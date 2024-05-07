from typing import Optional

import numpy as np
import torch
import torch.optim as optim


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


def tdoa_calibration_loss(
    sensor_positions, sound_positions, observed_lags, sr=96000
):
    """Error function for calibration of sensor positions using TDoA.
    To be used within a call to scipy.optimize.

    :param sensor_positions: sensor positions (this will be optimized)
    :param sound_positions: sound positions for each observed lag
    :param observed_lags: lags observed between sensors for each sound
    :param sr: sampling rate
    """
    # sp assumed in meters
    C = sensor_positions[0]
    sensor_positions = sensor_positions.reshape(-1, 3)
    error = 0.0
    for i, sound in enumerate(sound_positions):
        distances = (
            np.sqrt(np.sum((sound - sensor_positions) ** 2, axis=1)) / C
        )
        lags_samples = np.diff(distances) * sr
        # will have 1 - 0, 2 - 1 as lags in number of samples
        error += np.abs(lags_samples - observed_lags[i]) ** 1
    return np.mean(error)


def optimize_positions(
    observed_lags: torch.Tensor,
    initial_sensor_positions: torch.Tensor,
    initial_sound_positions: torch.Tensor,
    lr=0.01,
    num_epochs=1000,
    C=342.29,
    radius=0.1778,
    sr=96000,
    eps=1e-2,
    n_es=10,
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

    :return: Optimized sensor and sound positions
    """

    errors = []
    # Speed of sound in m/s
    C = torch.tensor(C, requires_grad=True, dtype=torch.float32)
    decay = torch.tensor(0.03, requires_grad=True, dtype=torch.float32)

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

    lrs = torch.tensor([1e-4, 1e-6, 1e-2, 1e-2], dtype=torch.float32) * lr
    optimizer = optim.Adam(
        [
            {"params": [sensor_positions], "lr": lrs[0]},
            {"params": [sp_learnable], "lr": lrs[1]},
            {"params": C, "lr": lrs[2]},
            {"params": decay, "lr": lrs[3]},
        ]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=list(lrs * 100), steps_per_epoch=1, epochs=num_epochs
    )
    errors.clear()
    last_loss = torch.inf
    counter = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Compute distances from each sound to each sensor
        sound_positions = torch.cat((sp_learnable, sp_nl), dim=1)
        distance_sound_sensor = torch.sqrt(
            torch.sum(
                (sound_positions[:, None, :] - sensor_positions[None, :, :])
                ** 2,
                dim=-1,
            )
        )
        # distances will be in seconds across each direction
        # distances = torch.sqrt(torch.sum(diff**2, dim=-1)) / C
        speed_adjustment = torch.exp(-distance_sound_sensor / decay)
        # speed_adjustment = torch.sigmoid(decay) * distances

        # The time it takes sound to travel the distance between sound source
        # and sensor position
        time_per_distance = distance_sound_sensor / (
            C * (1 + speed_adjustment)
        )
        # Compute lags in number of samples
        lags = torch.diff(time_per_distance) * sr
        error = torch.abs(lags - observed_lags) ** 1
        loss = error.mean()
        # Additional loss for max radius
        penalties = torch.relu(
            (sp_learnable**2).sum(dim=1) - (0.99 * radius) ** 2
        )
        # Crude early stopping on own training loss
        if loss < last_loss - eps:
            last_loss = loss
            counter = 0 if counter == 0 else counter - 1
        elif counter < n_es:
            counter += 1
        else:
            break
        loss += penalties.sum()
        errors.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        # print(C)
        scheduler.step()
        # Print progress
        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}, Loss {loss.item()}, LL"
                f" {last_loss.item() - eps}"
            )
    print(f"Epoch {epoch}, Loss {loss.item()}")
    return (
        sensor_positions.detach(),
        sound_positions.detach(),
        C.detach(),
        decay,
    )


# # Usage: Initialize your observed_lags, initial_sensor_positions, and
# # initial_sound_positions here Then call optimize_positions to get the
# # optimized positions
# mask = (mins == 2) | (mins == 5)

# # initial_sensor_positions = torch.tensor(
# #     initial_sensor_positions, dtype=torch.float32
# # )
# initial_sensor_positions = torch.tensor(
#     [*sensors[:2], snare_pos, sensors[2]],
#     dtype=torch.float32,
# )
# ns = len(initial_sensor_positions)
# tmins = mins[mask]
# tmins[tmins == 5] = 3
# tmins += torch.arange(0, ns * len(mins[mask]), ns)
# initial_sound_positions = torch.zeros(len(mins), 3)

# # Initialize likely snare hits at 0

# initial_sound_positions[mins == 2] = 0.0
# # initial_sound_positions[mins == 4] = initial_sensor_positions[4]
# initial_sound_positions[mins == 3] = initial_sensor_positions[3]
# # initial_sound_positions += (torch.rand(len(mins), 3) - 0.5) * 0.1
# senp, sonp = optimize_positions(
#     ol[mask][:, [0, 1, 2, 5]],
#     initial_sensor_positions,
#     initial_sound_positions[mask],
#     num_epochs=10000,
# )
