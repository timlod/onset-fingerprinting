from __future__ import annotations

from collections import OrderedDict

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.func import vmap
from torch.nn import functional as F

from onset_fingerprinting import plots


class GradProbe(L.Callback):
    """
    Print the first module whose *output* gradient is (almost) zero.

    Usage
    -----
    probe = GradProbe(tol=1e-10)        # tolerance for "zero"
    trainer = L.Trainer(callbacks=[probe, ...])
    """

    def __init__(self, *, tol: float = 0.0) -> None:
        super().__init__()
        self.tol = tol
        self._outs: OrderedDict[str, Tensor] = OrderedDict()

    # ------------------------------------------------------------------ #
    # register forward hooks on *leaf* modules                            #
    # ------------------------------------------------------------------ #
    def on_fit_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:

        def _hook(name: str):
            def _fwd(_: torch.nn.Module, __, out: Tensor) -> None:
                # keep only differentiable tensors
                if (
                    isinstance(out, Tensor)
                    and out.requires_grad
                    and out.dtype.is_floating_point
                ):
                    out.retain_grad()
                    self._outs[name] = out

            return _fwd

        for name, mod in pl_module.named_modules():
            if not list(mod.children()):  # leaf module
                mod.register_forward_hook(_hook(name))

    # ------------------------------------------------------------------ #
    # after backward: locate first zero-grad tensor                       #
    # ------------------------------------------------------------------ #
    def on_after_backward(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:

        for name, out in self._outs.items():
            g = out.grad
            mean = 0.0 if g is None else float(g.abs().mean())
            if mean <= self.tol:
                print(f"[GradProbe] ⟂ gradient at → {name}")
                break

        # free references each step
        self._outs.clear()


class TrilaterationSolver(nn.Module):
    """
    Batched Newton–Raphson solver for 2-D trilateration, differentiable w.r.t.
    all inputs.  Accepts tensors with leading batch dim B.

    It solves, for point p = (x, y):

    .. math::

        ||p-a|| - ||p-o|| = Δd_a
        ||p-b|| - ||p-o|| = Δd_b
    """

    def __init__(
        self,
        max_iter: int = 20,
        tol: float = 1e-2,
        eps: float = 1e-9,
        use_lstsq: bool = True,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.use_lstsq = use_lstsq

    def forward(
        self,
        sensor_a: Tensor,  # (B, 2)
        sensor_b: Tensor,  # (B, 2)
        sensor_origin: Tensor,  # (B, 2)
        delta_d_a: Tensor,  # (B,)
        delta_d_b: Tensor,  # (B,)
        initial_guess: Tensor,  # (B, 2)
    ) -> Tensor:  # (B, 2)
        """
        Parameters
        ----------
        sensor_a, sensor_b, sensor_origin
            Cartesian coordinates shaped ``(B, 2)``.
        delta_d_a, delta_d_b
            Signed range-difference measurements shaped ``(B,)``.
        initial_guess
            Initial position estimate shaped ``(B, 2)``.

        Returns
        -------
        Tensor
            Estimated positions shaped ``(B, 2)``.
        """
        p = initial_guess
        B = p.shape[0]

        for _ in range(self.max_iter):
            d_a = torch.norm(p - sensor_a, dim=-1).clamp_min(self.eps)
            d_b = torch.norm(p - sensor_b, dim=-1).clamp_min(self.eps)
            d_o = torch.norm(p - sensor_origin, dim=-1).clamp_min(self.eps)

            f1 = d_a - d_o - delta_d_a
            f2 = d_b - d_o - delta_d_b
            F = torch.stack((f1, f2), dim=-1)

            x, y = p.unbind(-1)
            x_a, y_a = sensor_a.unbind(-1)
            x_b, y_b = sensor_b.unbind(-1)
            x_o, y_o = sensor_origin.unbind(-1)

            j00 = (x - x_a) / d_a - (x - x_o) / d_o
            j01 = (y - y_a) / d_a - (y - y_o) / d_o
            j10 = (x - x_b) / d_b - (x - x_o) / d_o
            j11 = (y - y_b) / d_b - (y - y_o) / d_o

            J = torch.stack(
                (
                    torch.stack((j00, j01), dim=-1),
                    torch.stack((j10, j11), dim=-1),
                ),
                dim=-2,
            )  # (B, 2, 2)

            if self.use_lstsq:
                delta = torch.linalg.lstsq(
                    J, -F.unsqueeze(-1)
                ).solution.squeeze(-1)
            else:
                I = torch.eye(2, dtype=J.dtype, device=J.device).expand(
                    B, 2, 2
                )
                JT = J.transpose(-2, -1)
                H = JT @ J + 1e-6 * I
                g = JT @ F.unsqueeze(-1)
                delta = torch.linalg.solve(H, -g).squeeze(-1)

            p = p + delta

            if delta.abs().amax(dim=-1).max() < self.tol:
                break

        return p


class TrilaterationSolver(nn.Module):
    def __init__(
        self,
        max_iter: int = 20,
        tol: float = 1e-2,
        eps: float = 1e-6,
        use_lstsq: bool = True,
        lm_init: float = 1e-2,
        lm_decay: float = 0.3,
        lm_growth: float = 10.0,
        lm_tries: int = 3,
        rcond: float | None = 1e-6,
        use_float64_internal: bool = True,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.use_lstsq = use_lstsq
        self.lm_init = lm_init
        self.lm_decay = lm_decay
        self.lm_growth = lm_growth
        self.lm_tries = lm_tries
        self.rcond = rcond
        self.use_float64_internal = use_float64_internal

    def forward(
        self,
        sensor_a: torch.Tensor,  # (B, 2)
        sensor_b: torch.Tensor,  # (B, 2)
        sensor_origin: torch.Tensor,  # (B, 2)
        delta_d_a: torch.Tensor,  # (B,)
        delta_d_b: torch.Tensor,  # (B,)
        initial_guess: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:  # (B, 2)
        # Use higher precision internally for stability.
        if self.use_float64_internal:
            dtype = torch.float64
        else:
            dtype = initial_guess.dtype

        p = initial_guess.to(dtype)
        sa = (sensor_a - sensor_origin).to(dtype)
        sb = (sensor_b - sensor_origin).to(dtype)
        o = torch.zeros_like(sa)
        do_not_use = sensor_origin  # silence linter about unused var
        p = p - sensor_origin.to(dtype)

        B = p.shape[0]
        I2 = torch.eye(2, dtype=dtype, device=p.device).expand(B, 2, 2)

        # Residual builder for LM acceptance checks.
        def residual_at(pp: torch.Tensor) -> torch.Tensor:
            da = torch.norm(pp - sa, dim=-1).clamp_min(self.eps)
            db = torch.norm(pp - sb, dim=-1).clamp_min(self.eps)
            d0 = torch.norm(pp, dim=-1).clamp_min(self.eps)
            f1 = da - d0 - delta_d_a.to(dtype)
            f2 = db - d0 - delta_d_b.to(dtype)
            F = torch.stack((f1, f2), dim=-1)
            return (F * F).sum(dim=-1)

        for _ in range(self.max_iter):
            da = torch.norm(p - sa, dim=-1).clamp_min(self.eps)
            db = torch.norm(p - sb, dim=-1).clamp_min(self.eps)
            d0 = torch.norm(p, dim=-1).clamp_min(self.eps)

            f1 = da - d0 - delta_d_a.to(dtype)
            f2 = db - d0 - delta_d_b.to(dtype)
            F = torch.stack((f1, f2), dim=-1)  # (B, 2)

            x, y = p.unbind(-1)
            xa, ya = sa.unbind(-1)
            xb, yb = sb.unbind(-1)

            # Origin is (0, 0) in this frame.
            j00 = (x - xa) / da - x / d0
            j01 = (y - ya) / da - y / d0
            j10 = (x - xb) / db - x / d0
            j11 = (y - yb) / db - y / d0

            J = torch.stack(
                (
                    torch.stack((j00, j01), dim=-1),
                    torch.stack((j10, j11), dim=-1),
                ),
                dim=-2,
            )  # (B, 2, 2)

            # Initial residual norm for acceptance.
            res0 = (F * F).sum(dim=-1)

            # Levenberg–Marquardt via augmented least squares using lstsq.
            lam = torch.full((B,), self.lm_init, dtype=dtype, device=p.device)
            best_delta = torch.zeros(B, 2, dtype=dtype, device=p.device)
            best_res = res0.clone()

            for _try in range(self.lm_tries):
                lam_sqrt = lam.sqrt().view(B, 1, 1)
                A_aug = torch.cat((J, lam_sqrt * I2), dim=-2)  # (B, 4, 2)
                b_aug = torch.cat(
                    (
                        -F.unsqueeze(-1),
                        torch.zeros(B, 2, 1, dtype=dtype, device=p.device),
                    ),
                    dim=-2,
                )  # (B, 4, 1)
                try:
                    delta = torch.linalg.lstsq(A_aug, b_aug).solution
                    delta = delta.squeeze(-1)
                except RuntimeError:
                    # Final fallback: pinv with cutoff.
                    delta = -(
                        torch.linalg.pinv(J, rcond=self.rcond)
                        @ F.unsqueeze(-1)
                    ).squeeze(-1)

                p_try = p + delta
                res_try = residual_at(p_try)

                improved = res_try < best_res
                best_delta = torch.where(
                    improved.unsqueeze(-1), delta, best_delta
                )
                best_res = torch.where(improved, res_try, best_res)

                lam = torch.where(
                    improved, lam * self.lm_decay, lam * self.lm_growth
                )

            p = p + best_delta

            if best_delta.abs().amax(dim=-1).max() < self.tol:
                break

        # Return to original frame and dtype.
        p = p + sensor_origin.to(dtype)
        return p.to(initial_guess.dtype)


def paired_xcorr(
    x: torch.Tensor,
    C: int,
    K: int,
) -> torch.Tensor:
    """
    Cross-correlate every adjacent channel-pair (1&2, 2&3, …) in each
    feature-map using grouped conv1d.

    :param x: Tensor of shape (B, C*K, V)
    :param C: Channels per feature-map
    :param K: Number of feature-maps
    :return: Tensor of shape (B, (C-1)*K, 2*V - 1)
    """
    B, CK, V = x.shape
    assert CK == C * K

    # (B, C, K, V)
    x = x.view(B, C, K, V)

    # Extract adjacent pairs: (B, C-1, K, V)
    a = x[:, :-1, :, :]
    b = x[:, 1:, :, :]

    # Reshape to (B, (C-1)*K, V)
    a = a.view(B, (C - 1) * K, V)
    b = b.view(B, (C - 1) * K, V)

    M = B * (C - 1) * K

    a_pad = F.pad(a, (V - 1, V - 1)).view(1, M, 3 * V - 2)

    out = F.conv1d(a_pad, b.reshape(M, 1, V), groups=M)
    return out.view(B, (C - 1), K, 2 * V - 1).mean(dim=2)


## TODO: number of sample offsets as parameter to optimize
# TODO: Train a physical model later to do onset rejection, swap out final
# layer for that
## TODO: Add some other data in to classify a void zone
# TODO: OD on close mic, backtrack taking into account direction, wait N
# samples and trigger regression (or perhaps classification first)
class CNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        loss=F.l1_loss,
        batch_norm=False,
        pool=False,
        padding=1,
        dilation=1,
        groups=1,
        lr=1e-3,
        activation=nn.SiLU,
    ) -> None:
        """
        A flexible CNN architecture for audio processing tasks.

        :param window_size: The size of the 1D audio window for each sensor.
        :param output_size: The dimensionality of the output (e.g., 2D
            coordinates).
        :param channels: Number of input channels (sensors).
        :param conv_layers_config: List of dictionaries defining each
            convolutional layer configuration.
        :param dropout_rate: Dropout rate applied after all convolutional
            layers.
        """
        super().__init__()
        self.conv_layers = nn.Sequential()

        current_channels = channels
        # Input size to the first layer
        inp = torch.zeros(1, channels, input_size)
        for i, layer_size in enumerate(layer_sizes):
            conv = nn.Conv1d(
                in_channels=current_channels,
                out_channels=layer_size,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            inp = conv(inp)
            self.conv_layers.add_module(f"conv{i+1}", conv)
            self.conv_layers.add_module(f"act{i+1}", activation())
            if batch_norm:
                self.conv_layers.add_module(
                    f"bn{i+1}", nn.BatchNorm1d(layer_size)
                )
            if pool:
                mp = nn.MaxPool1d(kernel_size=2, stride=2)
                self.conv_layers.add_module(f"pool{i+1}", mp)
                inp = mp(inp)
            current_channels = layer_size

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels * inp.shape[-1], output_size)
        self.loss = loss
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("hp_metric", loss)
        plots.cartesian_circle(out.cpu().detach().numpy())
        self.logger.experiment.add_figure("test", plt.gcf())
        plt.close()
        return loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=100
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 250, 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        pass


class RNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.5,
        loss=F.l1_loss,
        rnn_type: str = "GRU",  # Options: 'LSTM', 'GRU', 'RNN'
        batch_first: bool = True,
        bidirectional: bool = False,
        bias: bool = True,
        lr: float = 1e-3,
        activation=nn.SiLU,
        num_heads: int = 2,
        share_input_weights: bool = False,
        permute_input: bool = True,
    ) -> None:
        """
        A flexible RNN architecture for audio processing tasks.

        :param input_size: The size of the 1D audio window for each sensor.
        :param output_size: The dimensionality of the output (e.g., 2D
            coordinates).
        :param channels: Number of input channels (sensors).
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param dropout_rate: Dropout rate applied after the RNN layers.
        :param rnn_type: Type of RNN ('LSTM', 'GRU', 'RNN').
        :param batch_first: If True, then the input and output tensors are
            provided as (batch, seq, feature).
        :param bidirectional: If True, becomes a bidirectional RNN.
        """
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lr = lr
        self.loss = loss
        self.share_input_weights = share_input_weights
        self.permute_input = permute_input

        rnn_class = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[rnn_type]

        self.rnn = rnn_class(
            input_size=channels if not share_input_weights else 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=bidirectional,
            bias=bias,
        )
        multiplier = 2 if bidirectional else 1
        multiplier *= 1 if not share_input_weights else channels - 1
        # If we share weights we will stack the outputs for each pair of
        # channels
        self.layer_norm = nn.LayerNorm(hidden_size * multiplier)
        self.attention = nn.MultiheadAttention(
            hidden_size * multiplier,
            num_heads,
            batch_first=True,
            dropout=dropout_rate,
        )
        # multiplier *= 1 if not share_input_weights else channels - 1
        self.fc = nn.Linear(hidden_size * multiplier, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute_input:
            x = x.permute(0, 2, 1)
        if not self.share_input_weights:
            out, _ = self.rnn(x)
            out = self.layer_norm(out)
            # out = out[:, -1, :]  # Take the output of the last time step
            out, _ = self.attention(out, out, out, need_weights=False)
        # else:
        #     outs = []
        #     for i in range(self.channels - 1):
        #         out = self.rnn(x[..., i : i + 2])[0]
        #         out = self.layer_norm(out)
        #         out, _ = self.attention(out, out, out, need_weights=False)
        #         outs.append(out)
        #     out = torch.cat(outs, dim=-1)
        else:
            outs = []
            for i in range(self.channels - 1):
                outs.append(self.rnn(x[..., i : i + 2])[0])
            out = torch.cat(outs, dim=-1)
            out = self.layer_norm(out)
            out, _ = self.attention(out, out, out, need_weights=False)
        out = self.fc(out.mean(1))
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("hp_metric", loss)
        plots.cartesian_circle(out.cpu().detach().numpy())
        self.logger.experiment.add_figure("test", plt.gcf())
        plt.close()
        return loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(
            self.parameters(), lr=self.lr, weight_decay=1e-4
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=100
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 3000)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 250, 1
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


class CNNRNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        n_hidden: int = 64,
        n_rnn_layers: int = 1,
        loss=F.l1_loss,
        batch_norm=False,
        pool=False,
        padding=1,
        dilation=1,
        groups=1,
        lr=1e-3,
        activation=nn.SiLU,
    ) -> None:
        """
        A flexible CNN architecture for audio processing tasks.

        :param window_size: The size of the 1D audio window for each sensor.
        :param output_size: The dimensionality of the output (e.g., 2D
            coordinates).
        :param channels: Number of input channels (sensors).
        :param conv_layers_config: List of dictionaries defining each
            convolutional layer configuration.
        :param dropout_rate: Dropout rate applied after all convolutional
            layers.
        """
        super().__init__()
        self.conv_layers = nn.Sequential()

        current_channels = channels
        # Input size to the first layer
        inp = torch.zeros(1, channels, input_size)
        for i, layer_size in enumerate(layer_sizes):
            conv = nn.Conv1d(
                in_channels=current_channels,
                out_channels=layer_size,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            inp = conv(inp)
            self.conv_layers.add_module(f"conv{i+1}", conv)
            self.conv_layers.add_module(f"act{i+1}", activation())
            if batch_norm:
                self.conv_layers.add_module(
                    f"bn{i+1}", nn.BatchNorm1d(layer_size)
                )
            if pool:
                mp = nn.MaxPool1d(kernel_size=2, stride=2)
                self.conv_layers.add_module(f"pool{i+1}", mp)
                inp = mp(inp)
            current_channels = layer_size

        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(
            inp.shape[2],
            n_hidden,
            n_rnn_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        inp, _ = self.rnn(inp)
        # self.fc = nn.Linear(rnn_hidden, output_size)

        self.attention = nn.MultiheadAttention(
            inp.shape[2], 2, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(inp.shape[2], output_size)
        inp = self.fc(inp)
        self.loss = loss
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        # x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x, _ = self.attention(x, x, x, need_weights=False)
        # Consider using all timesteps
        # x = self.fc(x[:, -1, :])
        x = self.fc(x.mean(1))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.l1_loss(out, y)
        self.log("hp_metric", loss)
        plots.cartesian_circle(out.cpu().detach().numpy())
        self.logger.experiment.add_figure("test", plt.gcf())
        plt.close()
        return loss

    def configure_optimizers(self):
        optimizer = optim.NAdam(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=100
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 250, 1
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


class CNN2(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
        dropout_rate: float = 0.5,
        batch_norm=False,
        pool=False,
        padding=1,
        dilation=1,
        group: bool = False,
        activation=nn.SiLU,
    ) -> None:
        """
        A flexible CNN architecture.

        :param input_size: The size of the 1D audio window for each sensor.
        :param output_size: The dimensionality of the output (e.g., 2D
            coordinates).
        :param channels: Number of input channels (sensors).
        """
        super().__init__()
        self.conv_layers = nn.Sequential()
        self.group = group
        self.channels = channels

        current_channels = channels if group else 1

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(layer_sizes)
        if isinstance(strides, int):
            strides = [strides] * len(layer_sizes)

        # Input size to the first layer
        inp = torch.zeros(1, current_channels, input_size)
        xs = [inp]
        for i, (layer_size, kernel_size, stride) in enumerate(
            zip(layer_sizes, kernel_sizes, strides)
        ):
            conv = nn.Conv1d(
                in_channels=current_channels,
                out_channels=layer_size * (channels if group else 1),
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                groups=channels if group else 1,
            )
            inp = conv(inp)
            self.conv_layers.add_module(f"conv{i+1}", conv)
            self.conv_layers.add_module(f"act{i+1}", activation())
            if batch_norm:
                self.conv_layers.add_module(
                    f"bn{i+1}",
                    # nn.BatchNorm1d(layer_size * (channels if group else 1)),
                    nn.GroupNorm(1, layer_size * (channels if group else 1)),
                )
            if pool:
                mp = nn.MaxPool1d(kernel_size=2, stride=2)
                self.conv_layers.add_module(f"pool{i+1}", mp)
                inp = mp(inp)
            current_channels = layer_size * (channels if group else 1)
            xs.append(inp)

        self.dropout = nn.Dropout(dropout_rate)
        output_dim = inp.shape[-1] + xs[1].shape[-1]
        # self.fc = nn.Linear((channels) * output_dim, output_size, bias=False)
        # self.fc = nn.Linear((channels - 1) * (output_dim), 3, bias=False)
        self.fc = nn.Linear(channels * (output_dim), 3, bias=False)

        self.fc = nn.Sequential(
            self.fc, nn.SiLU(), nn.Linear(3, output_size, bias=False)
        )
        print(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        if self.group:
            x = self.conv_layers(x)  # (B, C*K, V)
        else:
            # Unsqueeze to (C, B, 1, W) for vmap over channels
            # x = vmap(self.conv_layers, in_dims=1, out_dims=1)(x.unsqueeze(2))
            x = vmap(self.conv_layers[0], in_dims=1, out_dims=1)(
                x.unsqueeze(2)
            )
            x1 = x
            x = vmap(self.conv_layers[1:], in_dims=1, out_dims=1)(x)
            # → (B, C, K, V)
            x = x.reshape(B, C * x.shape[2], x.shape[3])
            # → (B, C*K, V)

        B, CK, V = x.shape
        K = CK // C
        x = x.view(B, C, K, V).mean(dim=2)
        x = torch.cat((x, x1.mean(dim=2)), dim=2)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class CCCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sensor_pos: torch.tensor,
        c: float = 82.0,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
        dropout_rate: float = 0.5,
        batch_norm: bool = False,
        pool: bool = False,
        padding: int = 1,
        dilation: int = 1,
        group: bool = False,
        activation=nn.SiLU,
    ) -> None:
        """
        A flexible CNN architecture to mimic computation of the
        cross-correlation (CC).

        :param input_size: The size of the 1D audio window for each sensor.
        :param output_size: The dimensionality of the output (e.g., 2D
            coordinates).
        :param channels: Number of input channels (sensors).
        """
        super().__init__()
        self.conv_layers = nn.Sequential()
        self.group = group
        self.channels = channels

        current_channels = channels if group else 1

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(layer_sizes)
        if isinstance(strides, int):
            strides = [strides] * len(layer_sizes)

        # Input size to the first layer
        inp = torch.zeros(1, current_channels, input_size)
        for i, (layer_size, kernel_size, stride) in enumerate(
            zip(layer_sizes, kernel_sizes, strides)
        ):
            conv = nn.Conv1d(
                in_channels=current_channels,
                out_channels=layer_size * (channels if group else 1),
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
                groups=channels if group else 1,
            )
            inp = conv(inp)
            self.conv_layers.add_module(f"conv{i+1}", conv)
            self.conv_layers.add_module(f"act{i+1}", activation())
            if batch_norm:
                self.conv_layers.add_module(
                    f"bn{i+1}",
                    # nn.BatchNorm1d(layer_size * (channels if group else 1)),
                    nn.GroupNorm(1, layer_size * (channels if group else 1)),
                )
            if pool:
                mp = nn.MaxPool1d(kernel_size=2, stride=2)
                self.conv_layers.add_module(f"pool{i+1}", mp)
                inp = mp(inp)
            current_channels = layer_size * (channels if group else 1)

        self.dropout = nn.Dropout(dropout_rate)
        output_dim = inp.shape[-1]
        self.fc = nn.Linear((channels) * output_dim, channels * 2, bias=False)
        # Potentially optimize too using parameter
        self.register_buffer("sensor_pos", sensor_pos)
        # self.register_parameter("sensor_pos", nn.Parameter(sensor_pos))
        self.register_buffer(
            "radius", torch.tensor(torch.sqrt(torch.sum(sensor_pos[0] ** 2)))
        )
        # self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        # self.register_parameter("sr", sr)
        self.solver = TrilaterationSolver(use_lstsq=True)
        print(self.fc, output_dim)

    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Forward call

        :param x: audio window after/around onset
        :param i: index of sensor which triggered detection
        """
        B, C, _ = x.shape
        if self.group:
            x = self.conv_layers(x)  # (B, C*K, V)
        else:
            x = vmap(self.conv_layers, in_dims=1, out_dims=1)(x.unsqueeze(2))
            x = x.reshape(B, C * x.shape[2], x.shape[3])  # (B, C*K, V)

        _, CK, V = x.shape
        K = CK // C
        x = x.view(B, C, K, V).mean(dim=2)
        x = torch.flatten(x, start_dim=1)
        lags = self.fc(x)  # .clip(-self.radius, self.radius)
        # lags = lags * self.radius
        # lags = (lags * self.radius).clip(-self.radius, self.radius)

        # Potentially use attention here to select lag from inter?
        # could use sr/c to decouple from sr and stuff like room temperature
        d_a, d_b = lags[range(B), 2 * i], lags[range(B), 2 * i + 1]
        # print(d_a)
        js = [(i - 1) % len(self.sensor_pos), (i + 1) % len(self.sensor_pos)]
        sensor_o = self.sensor_pos[i]
        sensor_a = self.sensor_pos[js[0]]
        sensor_b = self.sensor_pos[js[1]]

        weight_a = abs(d_a) / self.radius
        weight_b = abs(d_b) / self.radius
        weight_o = abs(d_a + d_b) / (2 * self.radius)
        weight_a = weight_b = weight_o = 0.333

        ig = torch.stack(
            [
                sensor_a[:, 0] * weight_a
                + sensor_b[:, 0] * weight_b
                + sensor_o[:, 0] * weight_o,
                sensor_a[:, 1] * weight_a
                + sensor_b[:, 1] * weight_b
                + sensor_o[:, 1] * weight_o,
            ],
            dim=1,
        )
        return self.solver(sensor_a, sensor_b, sensor_o, d_a, d_b, ig)


class LLCNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sensor_pos: torch.tensor,
        c: float = 82.0,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
        dropout_rate: float = 0.5,
        batch_norm=False,
        pool=False,
        padding=1,
        dilation=1,
        group: bool = False,
        activation=nn.SiLU,
        loss=F.l1_loss,
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.model = CCCNN(
            input_size,
            output_size,
            sensor_pos,
            c,
            channels,
            layer_sizes,
            kernel_sizes,
            strides,
            dropout_rate,
            batch_norm,
            pool,
            padding,
            dilation,
            group,
            activation,
        )
        self.lr = lr
        self.loss = loss
        self.save_hyperparameters()

    def forward(self, x, idx):
        return self.model(x, idx)

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        out = self.model(x, idx)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, idx = batch
        out = self.model(x, idx)
        loss = F.l1_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, idx = batch
        out = self.model(x, idx)
        loss = F.l1_loss(out, y)
        self.log("hp_metric", loss)
        plots.cartesian_circle(
            out.cpu().detach().numpy(), figsize=(6, 6), limit_axes=True
        )
        self.logger.experiment.add_figure("test", plt.gcf())
        plt.close()
        return loss

    def configure_optimizers(self):
        # optimizer = optim.NAdam(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr * 100,
            momentum=0.8,
            weight_decay=1e-5,
            # nesterov=True,
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=100
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 300, 2
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_train_end(self):
        # print(self.model.solver.c, self.model.solver.sensors)
        pass

    # def configure_gradient_clipping(
    #     self,
    #     optimizer,  # the current optimiser
    #     gradient_clip_val: float | None,
    #     gradient_clip_algorithm: str | None,
    # ) -> None:

    #     # clip the layer you care about
    #     torch.nn.utils.clip_grad_value_(
    #         self.model.fc.parameters(), clip_value=1.0
    #     )

    #     # optionally keep Lightning's built-in clipping for the rest
    #     # (set gradient_clip_val in Trainer to a small number or None)
    #     self.clip_gradients(
    #         optimizer,
    #         gradient_clip_val=gradient_clip_val,
    #         gradient_clip_algorithm=gradient_clip_algorithm,
    #     )


class LCNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        channels: int = 3,
        layer_sizes: list[int] = [8, 16],
        kernel_sizes: int | list[int] = 3,
        strides: int | list[int] = 1,
        dropout_rate: float = 0.5,
        batch_norm=False,
        pool=False,
        padding=1,
        dilation=1,
        group: bool = False,
        activation=nn.SiLU,
        loss=F.l1_loss,
        lr=1e-3,
    ) -> None:
        super().__init__()
        self.model = CNN2(
            input_size,
            output_size,
            channels,
            layer_sizes,
            kernel_sizes,
            strides,
            dropout_rate,
            batch_norm,
            pool,
            padding,
            dilation,
            group,
            activation,
        )
        self.lr = lr
        self.loss = loss
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.l1_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = F.l1_loss(out, y)
        self.log("hp_metric", loss)
        plots.cartesian_circle(out.cpu().detach().numpy(), figsize=(6, 6))
        self.logger.experiment.add_figure("test", plt.gcf())
        plt.close()
        return loss

    def configure_optimizers(self):
        # optimizer = optim.NAdam(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr * 100,
            momentum=0.8,
            weight_decay=1e-5,
            # nesterov=True,
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.5, patience=100
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 300, 2
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def on_train_end(self):
        print(self.model.solver.c, self.model.solver.sensors)

    def configure_gradient_clipping(
        self,
        optimizer,  # the current optimiser
        gradient_clip_val: float | None,
        gradient_clip_algorithm: str | None,
    ) -> None:

        # clip the layer you care about
        torch.nn.utils.clip_grad_value_(
            self.model.fc.parameters(), clip_value=1.0
        )

        # optionally keep Lightning's built-in clipping for the rest
        # (set gradient_clip_val in Trainer to a small number or None)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
