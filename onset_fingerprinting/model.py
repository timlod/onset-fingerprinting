import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from onset_fingerprinting import plots


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


class FCNN(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[int] = [10, 10, 10],
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = True,
        l2_reg: float = 0.0,
        eye_init=False,
        eye_noise_floor=0.01,
        bias=True,
    ) -> None:
        """
        Initialize a flexible network to translate scalar inputs into scalar
        outputs.

        :param input_size: Number of input features.
        :param output_size: Number of output features.
        :param hidden_layers: List of integers specifying the size of hidden
            layers.
        :param activation: Activation function ('relu', 'tanh', 'sigmoid',
            etc.).
        :param dropout: Dropout rate between layers (default 0.0 means no
            dropout).
        :param batch_norm: If True, add batch normalization after each hidden
            layer.
        :param l2_reg: L2 regularization parameter (default 0.0).
        """
        super(FCNN, self).__init__()

        self.l2_reg = l2_reg
        layers = []
        layer_sizes = [input_size] + hidden_layers

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias)
            if eye_init:
                self.init_eye_weights(layer, eye_noise_floor)
            layers.append(layer)

            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layer = nn.Linear(layer_sizes[-1], output_size, bias=bias)
        if eye_init:
            self.init_eye_weights(layer, eye_noise_floor)
        layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        :param x: Input tensor of shape (batch_size, input_size)
        :return: Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)

    def l2_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss if specified.

        :return: L2 regularization loss
        """
        if self.l2_reg == 0.0:
            return torch.tensor(0.0)

        l2_loss = torch.tensor(0.0)
        for param in self.parameters():
            l2_loss += torch.sum(param**2)

        return self.l2_reg * l2_loss

    def init_eye_weights(self, layer, noise_floor=0.001):
        perturbation = (
            torch.randn(layer.out_features, layer.in_features) * noise_floor
        )
        layer.weight.data = (
            torch.eye(layer.out_features, layer.in_features) + perturbation
        )

    def call_np(self, lags) -> np.ndarray:
        """
        Process individual pairs of lags and returns the prediction as a numpy
        array.

        :param lags: observed lags
        """
        with torch.no_grad():
            return self(torch.tensor([lags], dtype=torch.float32)).numpy()[0]

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
