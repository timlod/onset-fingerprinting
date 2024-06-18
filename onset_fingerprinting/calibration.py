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
    norm: int = 1,
    errors=None,
):
    """Error function for calibration of sensor positions using TDoA.  To be
    used within a call to scipy.optimize.

    :param sensor_positions: sensor positions (this will be optimized)
    :param sound_positions: sound positions for each observed lag
    :param observed_tdoa: lags observed between sensors for each sound
    :param C: speed of sound
    :param norm: 1 for MAE, 2 for MSE
    :param errors: list to save errors of each sound (can be used to filter out
        bad data)
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
        e = np.abs(tdoa - observed_tdoa[i]) ** norm
        error += e
        if errors is not None:
            errors.append(e)
    return np.mean(error)


def tdoa_calib_loss_jac(
    params: np.ndarray,
    sound_positions: np.ndarray,
    observed_tdoa: np.ndarray,
    C: float = 343.0,
    norm: int = 1,
    e=None,
):
    """Jacobian for tdoa_calib_loss."""
    sensor_positions = params.reshape(-1, 3)
    jac = np.zeros_like(params)
    for i, sound in enumerate(sound_positions):
        distances = (
            np.sqrt(np.sum((sound - sensor_positions) ** 2, axis=1)) / C
        )
        tdoa = np.diff(distances)
        error_term = tdoa - observed_tdoa[i]
        sign_error_term = np.sign(error_term)
        weighted_error_term = (
            sign_error_term
            if norm == 1
            else sign_error_term * (np.abs(error_term) ** (norm - 1))
        )

        for j in range(sensor_positions.shape[0]):
            if j > 0:
                d_error_d_pos_j = weighted_error_term[j - 1] * (
                    (sensor_positions[j] - sound) / (distances[j] * C)
                )
            if j < sensor_positions.shape[0] - 1:
                d_error_d_pos_j_minus_1 = -weighted_error_term[j] * (
                    (sensor_positions[j] - sound) / (distances[j] * C)
                )
                if j > 0:
                    d_error_d_pos_j += d_error_d_pos_j_minus_1
                else:
                    d_error_d_pos_j = d_error_d_pos_j_minus_1

            jac[j * 3 : (j + 1) * 3] += d_error_d_pos_j / len(sound_positions)

    return jac


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


def tdoa_calib_loss_with_sp_jac(
    params: np.ndarray,
    observed_tdoa: np.ndarray,
    n_lugs: int = 10,
    n_each: int = 4,
    center_hits: int = 4,
    norm=1,
    opt_c: bool = False,
    C: float = 343.0,
    e=None,
):
    """Jacobian for the TDoA calibration loss function."""
    sound_positions = [(0, 0, 0)] * center_hits + [
        multilateration.spherical_to_cartesian(*pos)
        for pos in calibration.calibration_locations(
            n_lugs, n_each, params[0], 0
        )
    ]
    if opt_c:
        C = params[1]

    sensor_positions = params[(1 + opt_c) :].reshape(-1, 3)
    jac = np.zeros_like(params)
    for i, sound in enumerate(sound_positions):
        distances = (
            np.sqrt(np.sum((sound - sensor_positions) ** 2, axis=1)) / C
        )
        tdoa = np.diff(distances)
        error_term = tdoa - observed_tdoa[i]
        sign_error_term = np.sign(error_term)
        weighted_error_term = (
            sign_error_term
            if norm == 1
            else sign_error_term * (np.abs(error_term) ** (norm - 1))
        )

        # Gradient w.r.t. sensor positions
        for j in range(sensor_positions.shape[0]):
            if j > 0:
                d_error_d_pos_j = weighted_error_term[j - 1] * (
                    (sensor_positions[j] - sound) / (distances[j] * C)
                )
            if j < sensor_positions.shape[0] - 1:
                d_error_d_pos_j_minus_1 = -weighted_error_term[j] * (
                    (sensor_positions[j] - sound) / (distances[j] * C)
                )
                if j > 0:
                    d_error_d_pos_j += d_error_d_pos_j_minus_1
                else:
                    d_error_d_pos_j = d_error_d_pos_j_minus_1

            jac[
                (1 + opt_c) + j * 3 : (1 + opt_c) + (j + 1) * 3
            ] += d_error_d_pos_j / len(sound_positions)

        # Gradient w.r.t. hit radius
        jac[0] += np.sum(weighted_error_term) / len(sound_positions)

        # Gradient w.r.t. speed of sound C (if optimized)
        if opt_c:
            d_error_d_c = -np.sum(
                weighted_error_term
                * np.diff(np.sum((sound - sensor_positions) ** 2, axis=1))
                / (C**2)
            )
            jac[1] += d_error_d_c / len(sound_positions) * 1000
            print(f"Gradient with respect to C: {jac[:3]}")
    return jac


def optimize_C(
    tdoa,
    n_lugs=10,
    n_each=4,
    center_hits=4,
    norm=1,
    C_range=(336, 345),
    initial_C=343.0,
    radius=14 * 2.54 / 100 / 2,
    hits_at=0.155,
    filter_errors_above=3,
    sound_positions=None,
    initial_sensor_positions=None,
    bounds=None,
    **kwargs,
):
    """Optimize both sensor positions and the speed of sound.

    :param tdoa: observed TDoA
    :param n_lugs: number of lugs on the drum
    :param n_each: number of hits at each lug
    :param center_hits: number of center hits preceding lug hits
    :param norm: 1 for MAE, 2 for MSE
    :param C_range: range of values to search for C
    :param initial_C: initial guess for C
    :param radius: drum radius
    :param hits_at: distance of hits from origin (something <1 * radius)
    :param filter_errors_above: removes hits with particularly large errors
        after a first pass with initial_C.  If optimizing for a membrane speed,
        it makes sense to call this function twice, first without filtering,
        and then again with the optimized C as the initial guess and filtering
    """
    errors = []

    if sound_positions is None:
        sound_positions = np.array(
            [(0, 0, 0)] * center_hits
            + [
                multilateration.spherical_to_cartesian(*pos)
                for pos in calibration.calibration_locations(
                    n_lugs, n_each, hits_at, 0
                )
            ]
        )

    if initial_sensor_positions is None:
        initial_sensor_positions = np.array(
            [
                multilateration.spherical_to_cartesian(*pos)
                for pos in np.array(
                    [
                        (0.9, 140, 75),
                        (0.9, 10, 55),
                        (hits_at, 100, 15),
                    ]
                )
            ]
        )
    if bounds is None:
        bounds = [(None, None), (None, None), (0, None)] * 2 + [
            (-radius, radius),
            (-radius, radius),
            (0, radius),
        ]
    result = optimize.minimize(
        tdoa_calib_loss,
        initial_sensor_positions.flatten(),
        args=(sound_positions, tdoa, initial_C, norm, errors),
        jac=tdoa_calib_loss_jac,
        method="TNC",
        bounds=bounds,
        options={"maxfun": 10000},
    )
    initial_sensor_positions = result.x
    errors1 = np.array(errors).sum(axis=1)
    med = np.median(errors1)
    good_idx = np.where(errors1 < filter_errors_above * med)[0]
    print(f"Removing {len(tdoa) - len(good_idx)} hits!")

    def objective(C):
        fun = optimize.minimize(
            tdoa_calib_loss,
            initial_sensor_positions,
            args=(sound_positions[good_idx], tdoa[good_idx], C, norm),
            jac=tdoa_calib_loss_jac,
            method="TNC",
            bounds=bounds,
            options={"maxfun": 1000},
        ).fun
        return fun

    res = optimize.minimize_scalar(objective, bounds=C_range, method="bounded")
    best_C = res.x
    final_result = optimize.minimize(
        tdoa_calib_loss,
        initial_sensor_positions,
        args=(sound_positions[good_idx], tdoa[good_idx], best_C, norm),
        jac=tdoa_calib_loss_jac,
        method="TNC",
        bounds=bounds,
        options={"maxfun": 100000},
    )
    return final_result.x.reshape(-1, 3), best_C


def calibrate(
    onsets: np.ndarray,
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
    """Calibration function for sensor positions given calibration hits.

    :param onsets: onset array of shape [n_onsets, n_channels]
    :param sr: sampling rate
    :param C: speed of sound
    :param diameter: diameter of drum
    :param n_lugs: number of lugs on the drum
    :param n_each: number of hits at each lug
    :param hits_at: where along the radius the calibration hits are in [0, 1]
    :param center_hits: number of center hits (0, 0) preceding the lug hits
    :param norm: 1 for MAE, 2 for MSE
    :param filter_errors_above: removes the hits with errors this times as
        large as the median error
    :param opt_c: if True, optimize the speed of sound as well
    """
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


class FCNN(nn.Module):
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


def optimize_positions(
    observed_lags: torch.Tensor,
    initial_sensor_positions: torch.Tensor,
    initial_sound_positions: torch.Tensor,
    lr: float = 0.01,
    lossfun=F.mse_loss,
    num_epochs: int = 1000,
    C: float = 342.29,
    sr: int = 96000,
    radius: float = 0.1778,
    eps: float = 1e-12,
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
    observed_tdoa = observed_lags / sr
    errors = []

    # Make sure data is on the same device
    device = observed_tdoa.device
    # Speed of sound in m/s
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32, device=device)
    sensor_positions = torch.tensor(
        initial_sensor_positions,
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    sp_learnable = torch.tensor(
        initial_sound_positions[:, :2],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    sp_nl = torch.zeros(
        len(sp_learnable), 1, dtype=torch.float32, device=device
    )

    lrs = torch.tensor([2e-3, 1e-4, 0.1], dtype=torch.float32) * lr
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
        # Difference in sound to sensor distances of two sensor pairs, in s
        # Time difference of arrival
        tdoa = (distances[:, :2] - distances[:, 2:]) / C
        loss = lossfun(tdoa, observed_tdoa)
        # Additional loss for max radius
        # penalties = torch.relu(
        #     (sp_learnable**2).sum(dim=1) - (0.99 * radius) ** 2
        # )
        # Crude early stopping on own training loss
        if loss < last_loss - eps:
            last_loss = loss
            counter = 0
        elif counter < patience:
            counter += 1
        else:
            break
        # loss += penalties.sum()
        errors.append(loss.detach().numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([sensor_positions, sp_learnable, C], 1)
        optimizer.step()
        scheduler.step()
        if epoch % print_every == 0:
            print(
                f"Epoch {epoch}, Loss {loss.item()}, LL"
                f" {last_loss.item() - eps}"
            )
    print(f"Epoch {epoch}, Loss {loss.item()}")
    if debug:
        print(tdoa[:10], "\n", observed_tdoa[:10])
    return (
        sensor_positions.detach(),
        sound_positions.detach(),
        C.detach(),
    )


def train_location_model(
    observed_lags: torch.Tensor,
    sound_positions: torch.Tensor,
    lr: float = 0.01,
    lossfun: Callable = F.l1_loss,
    num_epochs: int = 1000,
    eps: float = 1e-9,
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
        optimizer, num_epochs / 10
    )
    errors.clear()
    last_loss = torch.inf
    counter = 0
    best_model = model
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        pos = model(observed_lags)
        error = lossfun(pos, sound_positions[:, :2])
        errors.append(error.detach().numpy())
        loss = error.mean()
        # Crude early stopping on own training loss
        if loss < last_loss - eps:
            last_loss = loss
            best_model = model
            counter = 0
        elif counter < patience:
            counter += 1
        else:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
    print(f"Epoch {epoch}, Loss {loss.item()}")
    if debug:
        print(pos[:10], "\n", sound_positions[:10])
    return best_model, errors
