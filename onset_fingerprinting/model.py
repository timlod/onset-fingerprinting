import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


class CNN(L.LightningModule):
    def __init__(
        self,
        window_size: int,
        output_size: int,
        channels: int = 3,
        conv_layers_config: list[dict] = None,
        dropout_rate: float = 0.5,
        groups=1,
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

        if conv_layers_config is None:
            conv_layers_config = [
                {
                    "out_channels": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                },
                {
                    "out_channels": 32,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                },
                {
                    "out_channels": 64,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                },
            ]

        self.conv_layers = nn.Sequential()

        current_channels = channels
        # Input size to the first layer
        conv_output_size = window_size
        for idx, config in enumerate(conv_layers_config):
            self.conv_layers.add_module(
                f"conv{idx+1}",
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=config["out_channels"],
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"],
                    dilation=config["dilation"],
                    groups=groups,
                ),
            )
            self.conv_layers.add_module(f"relu{idx+1}", nn.ReLU())
            self.conv_layers.add_module(
                f"bn{idx+1}", nn.BatchNorm1d(config["out_channels"])
            )
            self.conv_layers.add_module(
                f"pool{idx+1}", nn.MaxPool1d(kernel_size=2, stride=2)
            )

            # Compute the output size after convolution
            effective_kernel_size = (config["kernel_size"] - 1) * config[
                "dilation"
            ] + 1
            conv_output_size = (
                conv_output_size
                + 2 * config["padding"]
                - effective_kernel_size
            ) // config["stride"] + 1

            # Calculate the output size after pooling
            conv_output_size = (conv_output_size - 2) // 2 + 1

            current_channels = config["out_channels"]

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(current_channels * conv_output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
