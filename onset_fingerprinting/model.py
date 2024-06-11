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
        dilation=0,
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
        lr: float = 1e-3,
        activation=nn.SiLU,
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

        rnn_class = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[rnn_type]

        self.rnn = rnn_class(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_size,
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_size,
        ).to(x.device)

        if isinstance(self.rnn, nn.LSTM):
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
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
