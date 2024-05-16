from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import optimize
from torch import nn

from onset_fingerprinting import calibration, multilateration


def tdoa_calib_loss(
    params: np.ndarray,
    sound_positions: np.ndarray,
    observed_tdoa: np.ndarray,
    C: float = 343.0,
    errors=None,
):
    """Error function for calibration of sensor positions using TDoA.
    To be used within a call to scipy.optimize.

    :param sensor_positions: sensor positions (this will be optimized)
    :param sound_positions: sound positions for each observed lag
    :param observed_tdoa: lags observed between sensors for each sound
    :param C: speed of sound
    """
    sensor_positions = params.reshape(-1, 3)
    error = 0.0
    if errors is not None:
        errors.clear()
    for i, sound in enumerate(sound_positions):
        distances = (
            np.sqrt(np.sum((sound - sensor_positions) ** 2, axis=1)) / C
        )
        tdoa = np.diff(distances)
        # will have 1 - 0, 2 - 1 as lags in tdoa
        e = np.abs(tdoa - observed_tdoa[i])
        error += e
        if errors is not None:
            errors.append(e)
    return np.mean(error)


def tdoa_calib_loss_with_sp(
    params: np.ndarray,
    observed_tdoa: np.ndarray,
    n_lugs: int = 10,
    n_each: int = 4,
    center_hits: int = 4,
    norm=1,
    opt_c: bool = False,
    C: float = 343.0,
    errors=None,
):
    """Error function for calibration of sensor positions using TDoA.  To be
    used within a call to scipy.optimize.

    :param params: parameters to optimize.  Flat array including hit radius,
        potentially c, and sensor positions
    :param observed_tdoa: lags observed between sensors for each sound
    :param n_lugs: number of lugs on the drum
    :param n_each: number of hits at each lug
    :param center_hits: number of center hits (assumed to be at the beginning
        of the data)
    :param norm: 1 for MAE, 2 for MSE
    :param opt_c: True to take C from params[1], which will be optimized in
        conjunction with hit radius and sensor positions
    """
    sound_positions = [(0, 0, 0)] * center_hits + [
        multilateration.spherical_to_cartesian(*pos)
        for pos in calibration.calibration_locations(
            n_lugs, n_each, params[0], 0
        )
    ]
    if opt_c:
        C = params[1]

    if errors is not None:
        errors.clear()
    sensor_positions = params[(1 + opt_c) :].reshape(-1, 3)
    error = 0.0
    for i, sound in enumerate(sound_positions):
        distances = (
            np.sqrt(np.sum((sound - sensor_positions) ** 2, axis=1)) / C
        )
        tdoa = np.diff(distances)
        # will have 1 - 0, 2 - 1 as lags in tdoa
        e = np.abs(tdoa - observed_tdoa[i]) ** norm
        if errors is not None:
            errors.append(e)
        error += e
    return np.mean(error)


def calibrate(
    onsets,
    sr: int = 96000,
    C: float = 343.0,
    diameter: float = 14 * 2.54,
    n_lugs: int = 10,
    n_each: int = 4,
    hits_at: int = 0.9,
    center_hits: int = 4,
    norm: int = 1,
    filter_errors_above: float = 2,
    opt_c: bool = False,
):
    errors = []
    radius = diameter / 2 / 100
    tdoa = np.diff(onsets) / sr

    sound_positions = [(0, 0, 0)] * center_hits + [
        multilateration.spherical_to_cartesian(*pos)
        for pos in calibration.calibration_locations(
            n_lugs, n_each, hits_at * radius, 0
        )
    ]

    initial_sensor_positions = np.array(
        [
            multilateration.spherical_to_cartesian(*pos)
            for pos in np.array(
                [
                    (0.9, 140, 75),
                    (0.9, 10, 55),
                    (radius, 100, 15),
                ]
            )
        ]
    )

    result = optimize.minimize(
        tdoa_calib_loss_with_sp,
        (
            [radius * hits_at]
            + ([C] if opt_c else [])
            + list(initial_sensor_positions.flatten())
        ),
        args=(tdoa, n_lugs, n_each, center_hits, norm, opt_c, C, errors),
        jac=tdoa_calib_loss_with_sp_jac,
        method="TNC",
        bounds=[(0.5 * radius, 1.1 * radius)]
        + ([(336.0, 345.0)] if opt_c else [])
        + [(None, None), (None, None), (0, None)] * 2
        + [(-radius, radius), (-radius, radius), (0, radius)],
        options={"maxfun": 10000},
    )
    r = result.x[0]
    if opt_c:
        C = result.x[1]
    print(r, C)
    sound_positions = np.array(
        [(0, 0, 0)] * center_hits
        + [
            multilateration.spherical_to_cartesian(*pos)
            for pos in calibration.calibration_locations(n_lugs, n_each, r, 0)
        ]
    )

    final_sensor_positions = result.x[1 + opt_c :].reshape(-1, 3)

    # Find error spikes to remove, assuming that in those cases we have bad
    # lags
    errors1 = np.array(errors).sum(axis=1)
    med = np.median(errors1)
    good_idx = np.where(errors1 < filter_errors_above * med)[0]
    print(good_idx.shape)
    print(f"Removing {len(tdoa) - len(good_idx)} hits!")

    errors = []
    # Retry
    result = optimize.minimize(
        tdoa_calib_loss,
        final_sensor_positions.flatten(),
        args=(sound_positions[good_idx], tdoa[good_idx], C),
        method="TNC",
        bounds=[(None, None), (None, None), (0, None)] * 2
        + [(-radius, radius), (-radius, radius), (0, radius)],
        options={"maxfun": 10000},
    )
    errors = np.array(errors)
    final_sensor_positions = result.x.reshape(-1, 3)
    return final_sensor_positions


def calibration_locations(
    n_lugs: int,
    n_each: int | list[int],
    radius: float,
    add_z: Optional[int] = None,
    clockwise: bool = False,
):
    """
    Make list of spherical coordinates for calibration hits close to drum lugs.

    :param n_lugs: number of lugs on the drum
    :param n_each: how many hits there are at each lug (can be a list if there
        is no onset data for some of the hits at a specific lug)
    :param radius: the radius at which the calibration hits happened
    :param add_z: Adds a third component
    """
    n = len(n_each) if isinstance(n_each, list) else 1
    angles = np.repeat(range(0, 360, int(360 / n_lugs)), n_each)
    if not clockwise:
        angles = 360 - angles
    if add_z is not None:
        assert isinstance(
            add_z, int
        ), f"add_z needs to be an integer! (given: {add_z} ({type(add_z)}))"
        return list(
            zip(
                np.repeat(np.repeat([radius] * n, n_each), n_lugs),
                angles,
                np.repeat(np.repeat([add_z] * n, n_each), n_lugs),
            )
        )
    else:
        return list(
            zip(
                np.repeat(np.repeat([radius] * n, n_each), n_lugs),
                angles,
            )
        )


def find_onset_groups(
    sample_indexes: list[int],
    channels: list[int],
    max_distance: int = 1000,
    min_channels: int = 3,
) -> np.ndarray | None:
    """
    Find groups of onsets based on sample distance and number of channels.
    Given onsets detected on calibration hits, these are potential candidates
    where an onset has been detected on each channel of interest close to each
    other.

    :param sample_indexes: List of onset sample indexes
    :param channels: List of channels corresponding to sample_indexes
    :param max_distance: Maximum allowable distance between first and last
        onset samples in a group
    :param min_channels: Minimum number of different channels required in a
        group

    :return: 2D numpy array, each row represents a group, or None if no groups
             are found
    """
    groups = []
    current_group = []
    max_channel = max(channels)

    for sample, channel in zip(sample_indexes, channels):
        if not current_group:
            current_group.append((sample, channel))
            continue

        if abs(sample - current_group[0][0]) <= max_distance:
            current_group.append((sample, channel))
        else:
            unique_channels = len(set(ch for _, ch in current_group))
            if unique_channels >= min_channels:
                group_array = np.full((max_channel + 1,), np.nan)
                for s, ch in current_group:
                    group_array[ch] = s
                groups.append(group_array)
            current_group = [(sample, channel)]

    # Check the last group
    unique_channels = len(set(ch for _, ch in current_group))
    if unique_channels >= min_channels:
        group_array = np.full((max_channel + 1,), np.nan)
        for s, ch in current_group:
            group_array[ch] = s
        groups.append(group_array)

    if groups:
        return np.array(groups, dtype=int)
    else:
        return None


class FCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[int] = [10, 10, 10],
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
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


def optimize_positions(
    observed_lags: torch.Tensor,
    initial_sensor_positions: torch.Tensor,
    initial_sound_positions: torch.Tensor,
    lr: float = 0.01,
    num_epochs: int = 1000,
    C: float = 342.29,
    sr: int = 96000,
    radius: float = 0.1778,
    eps: float = 1e-2,
    patience: float = 10,
    print_every=10,
    debug: bool = False,
):
    """
    Optimize the positions of sensors and sounds based on observed lags.

    :param observed_lags: Tensor of observed lags for each sound and sensor
        pair, shape (num_sounds, num_sensors, num_sensors)
    :param initial_sensor_positions: Initial sensor positions, shape
        (num_sensors, 3)
    :param initial_sound_positions: Initial sound positions, shape (num_sounds,
        3)
    :param lr: Learning rate for optimizer
    :param num_epochs: Number of epochs for optimization
    :param C: initial speed of sound
    :param sr: sampling rate
    :param radius: drum radius (additional loss commented out)
    :param patience: early stopping patience
    :param eps: epsilon for early stopping
    :param print_every: print loss every this many epochs
    :param debug: print some additional info
    """
    observed_lags = observed_lags / sr
    errors = []
    # Speed of sound in m/s
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32)

    # Make sure data is on the same device
    device = observed_lags.device
    initial_sensor_positions = initial_sensor_positions.to(device)
    initial_sound_positions = initial_sound_positions.to(device)

    sensor_positions = torch.tensor(
        initial_sensor_positions, requires_grad=True, dtype=torch.float32
    )
    sp_learnable = torch.tensor(
        initial_sound_positions[:, :2], requires_grad=True, dtype=torch.float32
    )
    sp_nl = torch.zeros(len(sp_learnable), 1)

    lrs = torch.tensor([2e-3, 1e-4, 1e-0], dtype=torch.float32) * lr
    optimizer = optim.Adam(
        [
            {"params": [sensor_positions], "lr": lrs[0]},
            {"params": [sp_learnable], "lr": lrs[1]},
            {"params": C, "lr": lrs[2]},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=list(lrs * 100), steps_per_epoch=1, epochs=num_epochs
    # )
    errors.clear()
    last_loss = torch.inf
    counter = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Compute distances from each sound to each sensor
        sound_positions = torch.cat((sp_learnable, sp_nl), dim=1)
        distances = torch.sqrt(
            torch.sum(
                (sound_positions[:, None, :] - sensor_positions[None, :, :])
                ** 2,
                dim=-1,
            )
        )
        # Difference in sound tosensor distances of two sensor pairs, in s
        # Time difference of arrival
        tdoa = (distances[:, 1:] - distances[:, :1]) / C
        error = torch.abs(tdoa - observed_lags) ** 2
        loss = error.mean()
        # Additional loss for max radius
        # penalties = torch.relu(
        #     (sp_learnable**2).sum(dim=1) - (0.99 * radius) ** 2
        # )
        # Crude early stopping on own training loss
        if loss < last_loss - eps:
            last_loss = loss
            counter = 0 if counter == 0 else counter - 1
        elif counter < patience:
            counter += 1
        else:
            break
        # loss += penalties.sum()
        errors.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % print_every == 0:
            print(
                f"Epoch {epoch}, Loss {loss.item()}, LL"
                f" {last_loss.item() - eps}"
            )
    print(f"Epoch {epoch}, Loss {loss.item()}")
    if debug:
        print(tdoa[:10], "\n", observed_lags[:10])
    return (
        sensor_positions.detach(),
        sound_positions.detach(),
        C.detach(),
    )


def train_location_model(
    observed_lags: torch.Tensor,
    sound_positions: torch.Tensor,
    lr: float = 0.01,
    loss: Callable = F.l1_loss,
    num_epochs: int = 1000,
    eps: float = 1e-2,
    patience: int = 10,
    print_every: int = 10,
    debug: bool = False,
    **kwargs,
):
    """
    Train an FCNN to predict sound locations based on observed lags between
    sensor pairs.

    :param observed_lags: Tensor of observed lags for sound and sensor pairs,
        shape (num_sounds, num_sensors - 1)
    :param sound_positions: sound/hit positions, shape (num_sounds, 3)
    :param lr: Learning rate for optimizer
    :param loss: loss function of form loss(input, target)
    :param num_epochs: Number of epochs for optimization
    :param patience: early stopping patience
    :param eps: epsilon for early stopping
    :param print_every: print loss every this many epochs
    :param debug: print some additional info
    :param kwargs: parameters for FCNN to control the model to train
    """

    errors = []

    # Make sure data is on the same device
    device = observed_lags.device
    sound_positions = sound_positions.to(device)
    model = FCNN(observed_lags.shape[1], 2, **kwargs)
    model = model.to(device)

    optimizer = optim.Adam([{"params": model.parameters(), "lr": lr}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs
    )
    errors.clear()
    last_loss = torch.inf
    counter = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pos = model(observed_lags)
        error = loss(pos, sound_positions[:, :2])
        errors.append(error.detach().numpy())
        loss = error.mean()
        # Crude early stopping on own training loss
        if loss < last_loss - eps:
            last_loss = loss
            counter = 0 if counter == 0 else counter - 1
        elif counter < patience:
            counter += 1
        else:
            break

        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
    print(f"Epoch {epoch}, Loss {loss.item()}")
    if debug:
        print(pos[:10], "\n", sound_positions[:10])
    return model, errors
