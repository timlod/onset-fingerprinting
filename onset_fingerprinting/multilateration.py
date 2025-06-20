from typing import Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import fsolve
from scipy.signal import find_peaks

from onset_fingerprinting import detection

TEMPERATURE = 20.0
HUMIDITY = 0.5
DIAMETER = 14 * 2.54
STRIKE_FORCE = 1.0
# speed in m/s of sound through drumhead membrane
C_drumhead = 82
# medium used in sound propagation equations (air or drumhead)
MEDIUM = "air"
ONSET_TOL = 50
NORM_CUTOFF = 10
lookaround = ONSET_TOL + NORM_CUTOFF


def speed_of_sound(
    scale: int = 1,
    temperature: float = TEMPERATURE,
    humidity: float = HUMIDITY,
    medium=MEDIUM,
) -> float:
    """Compute the speed of sound, default in m/s

    :param scale: change scale from m/s. mm/s would require scale=10
    :param temperature: temperature
    :param humidity: humidity
    :param medium: 'air' or 'drumhead'
    """
    if medium == "air":
        return scale * (331.3 + 0.606 * temperature) * (1 + 0.0124 * humidity)
    else:
        return scale * C_drumhead


def cartesian_to_polar(x: float, y: float, r: float = None):
    """Convert 2D cartesian coordinates to polar coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param r: radius unit-normalize returned radius
    """
    if r is None:
        r = np.sqrt(x**2 + y**2)
    else:
        r = np.sqrt(x**2 + y**2) / r

    phi_radians = np.arctan2(y, x)

    # Adjust theta to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)

    return r, np.degrees(phi_radians)


def polar_to_cartesian(r: float, phi: float):
    """Convert 2D polar coordinates to cartesian coordinates.

    :param r: radius
    :param phi: angle in degrees
    """
    phi_radians = np.radians(phi)

    x = r * np.cos(phi_radians)
    y = r * np.sin(phi_radians)
    return x, y


def spherical_to_cartesian(
    r: float,
    phi: float,
    theta: float,
) -> (float, float, float):
    """Convert 3D spherical coordinates to Cartesian coordinates.

    By default, x-y rotation moves clockwise and starts at y=0 (East); and x-z
    rotation starts at x=0 moving counter-clockwise (up).

    :param r: radius
    :param phi: angle in the x-y plane in degrees
    :param theta: angle in the x-z plane in degrees

    :return: Cartesian coordinates as (x, y, z)
    """
    phi_radians = np.radians(phi)
    if theta < 0:
        theta = -theta
    else:
        theta = 90 - theta
    theta_radians = np.radians(theta)

    x = r * np.cos(phi_radians) * np.sin(theta_radians)
    y = r * np.sin(phi_radians) * np.sin(theta_radians)
    z = r * np.cos(theta_radians)

    return x, y, z


def cartesian_to_spherical(x: float, y: float, z: float):
    """Convert 3D cartesian coordinates to spherical/polar coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi_radians = np.arctan2(y, x)
    theta_radians = np.arccos(z / r)

    # Adjust phi to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)
    theta = np.degrees(theta_radians)
    if theta < 0:
        theta = -theta
    else:
        theta = 90 - theta
    return r, np.degrees(phi_radians), theta


def cartesian_to_cylindrical(x: float, y: float, z: float, r: float = None):
    """Convert 3D cartesian coordinates to cylindrical coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param r: radius unit-normalize returned radius
    """
    if r is None:
        r = np.sqrt(x**2 + y**2)
    else:
        r = np.sqrt(x**2 + y**2) / r

    phi_radians = np.arctan2(y, x)

    # Adjust theta to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)

    return r, np.degrees(phi_radians), z


def cylindrical_to_cartesian(r: float, phi: float, z: float):
    """Convert 2D polar coordinates to cartesian coordinates.

    :param r: radius
    :param phi: angle in degrees
    """
    theta_radians = np.radians(phi)

    x = r * np.cos(theta_radians)
    y = r * np.sin(theta_radians)
    return x, y, z


def remove_seed(groups, group):
    seed_index = group[0][0]
    seed_onset = group[1][0]
    new_groups = []
    for group in groups:
        if not ((group[0][0] == seed_index) and (group[1][0] == seed_onset)):
            new_groups.append(group)
    return new_groups


def solve_trilateration(
    sensor_a: tuple[float, float],
    sensor_b: tuple[float, float],
    sensor_origin: tuple[float, float],
    delta_d_a: float,
    delta_d_b: float,
    initial_guess: np.ndarray,
) -> tuple[float, float]:
    """
    Solve the trilateration problem using fsolve.
    """
    x_o, y_o = sensor_origin
    x_a, y_a = sensor_a
    x_b, y_b = sensor_b

    def equations(point: np.ndarray) -> np.ndarray:
        x, y = point
        d_a = np.sqrt((x - x_a) ** 2 + (y - y_a) ** 2)
        d_b = np.sqrt((x - x_b) ** 2 + (y - y_b) ** 2)
        d_o = np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2)

        eq1 = d_a - d_o - delta_d_a
        eq2 = d_b - d_o - delta_d_b

        return eq1, eq2

    def jacobian(point: np.ndarray) -> np.ndarray:
        x, y = point
        J00 = (x - x_a) / np.sqrt((x - x_a) ** 2 + (y - y_a) ** 2) - (
            x - x_o
        ) / np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2)
        J01 = (y - y_a) / np.sqrt((x - x_a) ** 2 + (y - y_a) ** 2) - (
            y - y_o
        ) / np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2)
        J10 = (x - x_b) / np.sqrt((x - x_b) ** 2 + (y - y_b) ** 2) - (
            x - x_o
        ) / np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2)
        J11 = (y - y_b) / np.sqrt((x - x_b) ** 2 + (y - y_b) ** 2) - (
            y - y_o
        ) / np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2)
        return [[J00, J01], [J10, J11]]

    root, info, ier, msg = fsolve(
        equations,
        initial_guess,
        full_output=True,
        xtol=0.01,
        maxfev=20,
        fprime=jacobian,
    )

    if ier == 1:
        return tuple(root)
    else:
        # print(
        #     f"Solving ({sensor_origin}, {sensor_a}, {sensor_b} failed: {msg}"
        # )
        return None


def solve_trilateration_3d(
    sensor_a: tuple[float, float, float],
    sensor_b: tuple[float, float, float],
    sensor_origin: tuple[float, float, float],
    delta_d_a: float,
    delta_d_b: float,
    initial_guess: np.ndarray,
) -> tuple[float, float] | None:
    """
    Solve the trilateration problem for a 2D point using 3D sensor locations.

    :param sensor_a: The (x, y, z) coordinates of sensor A.
    :param sensor_b: The (x, y, z) coordinates of sensor B.
    :param sensor_origin: The (x, y, z) coordinates of the origin sensor.
    :param delta_d_a: The difference in distance from the unknown point to
        sensor A and the origin sensor.
    :param delta_d_b: The difference in distance from the unknown point to
        sensor B and the origin sensor.
    :param initial_guess: An initial guess of the (x, y) position of the
        unknown point.

    :return: The (x, y) coordinates of the unknown point if solution is found,
             else None.
    """
    # Extract 3D coordinates
    x_o, y_o, z_o = sensor_origin
    x_a, y_a, z_a = sensor_a
    x_b, y_b, z_b = sensor_b

    def equations(point: np.ndarray) -> np.ndarray:
        x, y = point
        # Assume the z-coordinate for the estimated point is at the same height
        # as the origin
        z = 0.0

        # Calculate distances in 3D space but project the result into 2D
        d_a = np.sqrt((x - x_a) ** 2 + (y - y_a) ** 2 + (z - z_a) ** 2)
        d_b = np.sqrt((x - x_b) ** 2 + (y - y_b) ** 2 + (z - z_b) ** 2)
        d_o = np.sqrt((x - x_o) ** 2 + (y - y_o) ** 2 + (z - z_o) ** 2)

        eq1 = d_a - d_o - delta_d_a
        eq2 = d_b - d_o - delta_d_b

        return np.array([eq1, eq2])

    def jacobian(point: np.ndarray) -> np.ndarray:
        x, y = point
        # Again, use a fixed z-coordinate
        z = 0.0

        # Jacobian only with respect to x and y
        J00 = (x - x_a) / np.sqrt(
            (x - x_a) ** 2 + (y - y_a) ** 2 + (z - z_a) ** 2
        ) - (x - x_o) / np.sqrt(
            (x - x_o) ** 2 + (y - y_o) ** 2 + (z - z_o) ** 2
        )
        J01 = (y - y_a) / np.sqrt(
            (x - x_a) ** 2 + (y - y_a) ** 2 + (z - z_a) ** 2
        ) - (y - y_o) / np.sqrt(
            (x - x_o) ** 2 + (y - y_o) ** 2 + (z - z_o) ** 2
        )
        J10 = (x - x_b) / np.sqrt(
            (x - x_b) ** 2 + (y - y_b) ** 2 + (z - z_b) ** 2
        ) - (x - x_o) / np.sqrt(
            (x - x_o) ** 2 + (y - y_o) ** 2 + (z - z_o) ** 2
        )
        J11 = (y - y_b) / np.sqrt(
            (x - x_b) ** 2 + (y - y_b) ** 2 + (z - z_b) ** 2
        ) - (y - y_o) / np.sqrt(
            (x - x_o) ** 2 + (y - y_o) ** 2 + (z - z_o) ** 2
        )

        return np.array([[J00, J01], [J10, J11]])

    root, info, ier, msg = fsolve(
        equations,
        initial_guess,
        full_output=True,
        xtol=0.01,
        maxfev=20,
        fprime=jacobian,
    )

    if ier == 1:
        return tuple(root)
    else:
        return None


class Multilaterate3D:
    def __init__(
        self,
        sensor_locations: list[tuple[float, float, float]],
        drum_diameter: float = DIAMETER,
        medium: str = "drumhead",
        sr: int = 44100,
        c: float | None = None,
        model=None,
    ):
        """Initialize multilateration onset locator.

        :param sensor_locations: list of sensor location tuples in relative
            polar coordinates, with first index of the location tuple being the
            radius (usually in [0, 1], although e.g. a microphone could be
            placed outside of the drumhead, i.e. 1.1) and the second being the
            angle in degrees, with 0/360 being the top/north location of the
            drum.

            For example, [(0.9, 0), (0.9, 90)] defines two sensors placed
            towards the outer edge of the drum, at the north and east locations
            of the drumhead

        :param drum_diameter: diameter in cm of the drum
        :param medium: 'drumhead' for vibration/optical sensors, 'air' for
            microphones
        :param sr: sampling rate
        :param c: speed of sound in m/s.  uses speed_of_sound if not provided
        :param model: bypass trilateration by optimization, using this pytorch
            model for trilateration instead
        """
        self.c = speed_of_sound(100, medium=medium) if c is None else c * 100
        self.model = model
        if model is not None:
            self.model.eval()
        self.radius = drum_diameter / 2
        self.sensor_locs = [
            spherical_to_cartesian(x[0] * self.radius, x[1], x[2])
            for x in sensor_locations
        ]
        self.medium = medium
        self.sr = sr
        self.samples_per_cm = sr / self.c

        # Create small lag maps (centimeter resolution) just to determine
        # whether a given lag is feasible, quickly
        self.lag_maps = [{} for _ in range(len(self.sensor_locs))]
        # Max valid lags for sensor pairings
        self.max_lags = [{} for _ in range(len(self.sensor_locs))]
        # Min valid lags for sensor pairings
        self.min_lags = [{} for _ in range(len(self.sensor_locs))]
        for i in range(len(self.sensor_locs)):
            for j in range(len(self.sensor_locs)):
                if i == j:
                    continue
                lm = lag_map_3d(
                    self.sensor_locs[j],
                    self.sensor_locs[i],
                    d=drum_diameter,
                    sr=sr,
                    scale=1,
                    medium=self.medium,
                    # 2cm tolerance around edge of drum
                    tol=2,
                    c=self.c,
                )
                # Allow some negative values, specifically to allow some slack
                # in the the center region if sensors are placed circularly
                lm[lm < -self.samples_per_cm * 1] = np.nan
                self.lag_maps[i][j] = lm
                self.max_lags[i][j] = np.nanmax(lm)
                self.min_lags[i][j] = np.nanmin(lm)
        # Max valid lag (total) per sensor, to use in stopping condition
        self.max_max_lags = [
            np.nanmax(list(d.values())) for d in self.max_lags
        ]
        self.ongoing = []

    def is_legal(self, first_sensor: int, later_sensor: int, lag: int) -> bool:
        """Verifies that the given sensor/onset index combinations fall on the
        playing surface.

        :param first_sensor: index of the sensor within self.sensor_locs with
            the earlier onset
        :param later_sensor: index of the sensor within self.sensor_locs with
            the later onset
        :param lag: onset index of the sensor with the later onset
        """
        return (
            self.min_lags[first_sensor][later_sensor]
            < lag
            < self.max_lags[first_sensor][later_sensor]
        )

    def is_legal_3d(self, group, tolerance=1):
        # We take a tolerance of Xcm around the target - fine location is done
        # during fsolve trilateration
        tolerance *= self.samples_per_cm
        sensors, onsets = group[0], group[1]
        lag1 = onsets[1] - onsets[0]
        lag2 = onsets[2] - onsets[0]
        lm1 = self.lag_maps[sensors[0]][sensors[1]]
        lm2 = self.lag_maps[sensors[0]][sensors[2]]

        legal = (lm1 < lag1 + tolerance) & (lm1 > lag1 - tolerance)
        legal &= (lm2 < lag2 + tolerance) & (lm2 > lag2 - tolerance)
        res = np.unravel_index(np.argmax(legal > 0), legal.shape, "F")
        return res

    def locate(
        self,
        sensor_index: int,
        onset_index: int,
        rec_audio: Optional[detection.CircularArray] = None,
    ) -> None | tuple[float, float]:
        new_groups = []

        for group in self.ongoing:
            # Group: ([sensor_indexes], [onset_indexes])
            # Changing from group[1][0] - comparison always to first sensor
            lag = onset_index - group[1][0]
            if lag > self.max_max_lags[group[0][0]]:
                continue
            # If an adjustment moved an onset behind the next we have to swap
            if lag < 0:
                print(f"swapping: {lag=} {group=}")
                inter = (group[0][0], group[1][0])
                group[0][0] = sensor_index
                group[1][0] = onset_index
                sensor_index, onset_index = inter
                lag = -lag
            # print(f"{onset_index=}, {group=}, {lag=}")
            # TODO: Will have to check maximum lag here
            if sensor_index not in group[0]:
                # TODO: what to do if we need more samples into the future?
                # this is the case where we get a detection towards the end of
                # the buffer
                # also check whether we should be always ccing wrt first onset
                if rec_audio is not None:
                    # take audio between last two onsets
                    # actually first now - otherwise change to group[1][-1]
                    last_onset = group[1][0]
                    # look further back for CC
                    i = rec_audio.counter - last_onset + lookaround
                    # indexing one more because of diff, maybe there's a better
                    section = rec_audio[-i - 1 :][
                        :, [group[0][0], sensor_index]
                    ]
                    # print(
                    #     f"{last_onset=} {rec_audio.counter=} {i=} {section.shape=}"
                    # )
                    section = np.diff(
                        median_filter(section, 5, axes=0), axis=0
                    )
                    section[section >= 0] = 0
                    section = abs(section)
                    section_og = np.array([last_onset, onset_index]) - (
                        last_onset - lookaround
                    )
                    # Problem: possible to move onset beyond boundary
                    ## TODO: perhaps wait for next frame to process this?
                    new_lag = detection.cross_correlation_lag(
                        section[:, 0],
                        section[:, 1],
                        # if using latest, change to -1
                        onsets=(group[1][0], onset_index),
                        d=0,
                        onset_tolerance=ONSET_TOL,
                        normalization_cutoff=NORM_CUTOFF,
                    )
                    # print(f"{new_lag=}")
                    if new_lag is not None:
                        lag = new_lag
                        co, cn = detection.adjust_onset(
                            section_og,
                            section[:, 0],
                            section[:, 1],
                            lag,
                        )
                        # print(f"{co=}, {cn=}")
                        # again, use -1 here if last onset instead of first
                        group[1][0] += co
                        onset_index += cn
                if self.is_legal(group[0][0], sensor_index, lag):
                    group = (
                        group[0] + [sensor_index],
                        group[1] + [onset_index],
                    )
                    if len(group[0]) == 3:
                        # TODO: INVESTIGATE THIS
                        if group[0][0] == group[0][1]:
                            break
                        res = self.is_legal_3d(group)
                        if res != (0, 0):
                            # res is the index into the lag map, subtracting
                            # the radius brings us approximately into the
                            # coordinate system
                            res = np.array(res) - self.radius
                            # print(group, res, res == (0, 0))
                            # Should we try again if trilaterate fails? Should
                            # we purge everything with the same first element
                            # if it succeeds?
                            res = self.trilaterate(group, initial_guess=res)
                            # print(f"{res=} {group=}")
                            if res is not None:
                                new_groups = remove_seed(new_groups, group)
                            self.ongoing = new_groups
                            return res
                    new_groups.append(group)
            # Not reached maximum possible lag (and didn't return during
            # trilaterate), so keep this group for now
            if lag <= self.max_max_lags[group[0][0]]:
                new_groups.append(group)
        new_groups.append(([sensor_index], [onset_index]))
        self.ongoing = new_groups
        return None

    def trilaterate(
        self, group: tuple[list[int], list[int]], initial_guess
    ) -> tuple[float, float]:
        sensors, onsets = group[0], group[1]
        # TODO: don't bake in assumptions about order
        # best just always use a list where the index tells which index it is
        if sensors[1] == 1:
            sensors[1:] = [0, 1]
            onsets[1:] = onsets[2:0:-1]
        sensor_a = self.sensor_locs[sensors[1]]
        sensor_b = self.sensor_locs[sensors[2]]
        sensor_origin = self.sensor_locs[sensors[0]]

        # onset diff = number of sample difference in sound arrival. / sr alone
        # would be seconds (tdoa), * c is mm, tdoa expressed in distance
        # sound travels in the time it took for sound to arrive at two sensors

        d_a1 = onsets[1] - onsets[0]
        d_b1 = onsets[2] - onsets[0]
        if self.model is not None:
            # Our scale is in centimeters, hence *100
            res = self.model.call_np((d_a1, d_b1)) * 100
        else:
            res = solve_trilateration_3d(
                sensor_a,
                sensor_b,
                sensor_origin,
                d_a1 / self.sr * self.c,
                d_b1 / self.sr * self.c,
                initial_guess,
            )
            # print(
            #     f"{sensors=}, {onsets=} | {d_a1=}, {d_b1=} |"
            #     f" {initial_guess=} | {res=}"
            # )
        if res is not None:
            # return cartesian_to_polar(*res, self.radius)
            return res
        else:
            return None


class Multilaterate:
    def __init__(
        self,
        sensor_locations: list[tuple[float, float]],
        drum_diameter: float = DIAMETER,
        medium: str = "drumhead",
        sr: int = 44100,
    ):
        """Initialize multilateration onset locator.

        :param sensor_locations: list of sensor location tuples in relative
            polar coordinates, with first index of the location tuple being the
            radius (usually in [0, 1], although e.g. a microphone could be
            placed outside of the drumhead, i.e. 1.1) and the second being the
            angle in degrees, with 0/360 being the top/north location of the
            drum.

            For example, [(0.9, 0), (0.9, 90)] defines two sensors placed
            towards the outer edge of the drum, at the north and east locations
            of the drumhead

        :param drum_diameter: diameter in cm of the drum
        :param medium: 'drumhead' for vibration/optical sensors, 'air' for
            microphones
        :param sr: sampling rate
        """
        self.radius = drum_diameter / 2
        self.sensor_locs = [
            polar_to_cartesian(x[0] * self.radius, x[1])
            for x in sensor_locations
        ]
        self.medium = medium
        self.sr = sr
        self.samples_per_cm = sr / speed_of_sound(100, medium=medium)

        # Create small lag maps (centimeter resolution) just to determine
        # whether a given lag is feasible, quickly
        self.lag_maps = [{} for _ in range(len(self.sensor_locs))]
        # Max valid lags for sensor pairings
        self.max_lags = [{} for _ in range(len(self.sensor_locs))]
        # Min valid lags for sensor pairings
        self.min_lags = [{} for _ in range(len(self.sensor_locs))]
        for i in range(len(self.sensor_locs)):
            for j in range(len(self.sensor_locs)):
                if i == j:
                    continue
                lm = lag_map_2d(
                    self.sensor_locs[j],
                    self.sensor_locs[i],
                    d=drum_diameter,
                    sr=sr,
                    scale=1,
                    medium=self.medium,
                    # 2cm tolerance around edge of drum
                    tol=2,
                )
                # Allow some negative values, specifically to allow some slack
                # in the the center region if sensors are placed circularly
                lm[lm < -self.samples_per_cm * 1] = np.nan
                self.lag_maps[i][j] = lm
                self.max_lags[i][j] = np.nanmax(lm)
                self.min_lags[i][j] = np.nanmin(lm)
        # Max valid lag (total) per sensor, to use in stopping condition
        self.max_max_lags = [
            np.nanmax(list(d.values())) for d in self.max_lags
        ]
        self.ongoing = []

    def is_legal(self, first_sensor: int, later_sensor: int, lag: int) -> bool:
        """Verifies that the given sensor/onset index combinations fall on the
        playing surface.

        :param first_sensor: index of the sensor within self.sensor_locs with
            the earlier onset
        :param later_sensor: index of the sensor within self.sensor_locs with
            the later onset
        :param lag: onset index of the sensor with the later onset
        """
        return (
            self.min_lags[first_sensor][later_sensor]
            < lag
            < self.max_lags[first_sensor][later_sensor]
        )

    def is_legal_3d(self, group, tolerance=1):
        # We take a tolerance of Xcm around the target - fine location is done
        # during fsolve trilateration
        tolerance *= self.samples_per_cm
        sensors, onsets = group[0], group[1]
        lag1 = onsets[1] - onsets[0]
        lag2 = onsets[2] - onsets[0]
        lm1 = self.lag_maps[sensors[0]][sensors[1]]
        lm2 = self.lag_maps[sensors[0]][sensors[2]]

        legal = (lm1 < lag1 + tolerance) & (lm1 > lag1 - tolerance)
        legal &= (lm2 < lag2 + tolerance) & (lm2 > lag2 - tolerance)
        res = np.unravel_index(np.argmax(legal > 0), legal.shape, "F")
        return res

    def locate(
        self, sensor_index: int, onset_index: int
    ) -> None | tuple[float, float]:
        new_groups = []

        for group in self.ongoing:
            # Group: ([sensor_indexes], [onset_indexes])
            lag = onset_index - group[1][0]
            if sensor_index not in group[0]:
                if self.is_legal(group[0][0], sensor_index, lag):
                    group = (
                        group[0] + [sensor_index],
                        group[1] + [onset_index],
                    )
                    if len(group[0]) == 3:
                        res = self.is_legal_3d(group)
                        if res != (0, 0):
                            res = np.array(res) - self.radius
                            # print(group, res, res == (0, 0))
                            # Should we try again if trilaterate fails? Should
                            # we purge everything with the same first element
                            # if it succeeds? print(group, sensor_index,
                            # onset_index, lag)
                            res = self.trilaterate(group, res)
                            # if res is not None:
                            #     new_groups = remove_seed(new_groups, group)
                            self.ongoing = new_groups
                            return res
                    new_groups.append(group)
            # Not reached maximum possible lag (and didn't return during
            # trilaterate), so keep this group for now
            if lag <= self.max_max_lags[group[0][0]]:
                new_groups.append(group)
        new_groups.append(([sensor_index], [onset_index]))
        self.ongoing = new_groups
        return None

    def trilaterate(
        self, group: tuple[list[int], list[int]], initial_guess
    ) -> tuple[float, float]:
        sensors, onsets = group[0], group[1]
        sensor_a = self.sensor_locs[sensors[1]]
        sensor_b = self.sensor_locs[sensors[2]]
        sensor_origin = self.sensor_locs[sensors[0]]

        c = speed_of_sound(100, medium=self.medium)

        d_a1 = (onsets[1] - onsets[0]) * c / self.sr
        d_b1 = (onsets[2] - onsets[0]) * c / self.sr

        res = solve_trilateration(
            sensor_a, sensor_b, sensor_origin, d_a1, d_b1, initial_guess
        )
        if res is not None:
            return cartesian_to_polar(*res, self.radius)
        else:
            return None


class MultilateratePaired:
    def __init__(
        self,
        sensor_locations: list[tuple[float, float]],
        drum_diameter: float = DIAMETER,
        scale: float = 10,
        medium: str = "drumhead",
        sr: int = 44100,
    ):
        """Initialize multilateration onset locator.

        This pre-computes maps of theoretical arrival time offsets ('lags')
        between different sensors placed on a drum given the sensor locations,
        the drums diameter and a model of how quickly a drum onset sound should
        travel from the onset location towards each sensor.

        Given 'live' onset sample data of each sensor, or lags in number of
        samples between when an onset sound reached each sensor, it returns the
        most likely strike location of the onset.  It does this by simply
        checking which locations in each lag map are potential matches, and
        adding up matches of each map.  The index with the highest value (most
        matches) will be the predicted location.

        :param sensor_locations: list of sensor location tuples in relative
            polar coordinates, with first index of the location tuple being the
            radius (usually in [0, 1], although e.g. a microphone could be
            placed outside of the drumhead, i.e. 1.1) and the second being the
            angle in degrees, with 0/360 being the top/north location of the
            drum.

            For example, [(0.9, 0), (0.9, 90)] defines two sensors placed
            towards the outer edge of the drum, at the north and east locations
            of the drumhead

        :param drum_diameter: diameter in cm of the drum
        :param scale: scale to use for accuracy of results.  scale 1 would use
            a centimeter grid, default uses millimeters.
        :param medium: 'drumhead' for vibration/optical sensors, 'air' for
            microphones
        :param sr: sampling rate
        """
        self.radius = int(np.round(drum_diameter * scale / 2, 1))
        self.sensor_locs = [
            polar_to_cartesian(x[0] * self.radius, x[1])
            for x in sensor_locations
        ]
        self.scale = scale
        self.medium = medium
        self.sr = sr

        self.lag_maps = [{} for _ in range(len(self.sensor_locs))]
        for i in range(len(self.sensor_locs)):
            for k in [-1, 1]:
                # Wrap around to first/last mic_loc
                j = (i + k) % len(self.sensor_locs)
                self.lag_maps[i][j] = lag_map_2d(
                    self.sensor_locs[i],
                    self.sensor_locs[j],
                    d=drum_diameter,
                    sr=sr,
                    scale=scale,
                    medium="drumhead",
                )
        # Pre-allocating results array - possibly useless optimization
        self.res = np.zeros_like(self.lag_maps[0][1])

    def locate(self, lags: list[int], i: int) -> tuple[float, float]:
        js = [(i - 1) % len(self.sensor_locs), (i + 1) % len(self.sensor_locs)]
        sensor_a = self.sensor_locs[js[0]]
        sensor_b = self.sensor_locs[js[1]]
        sensor_origin = self.sensor_locs[i]

        c = speed_of_sound(100 * self.scale, medium=self.medium)

        d_a1 = lags[0] * c / self.sr
        d_b1 = lags[1] * c / self.sr

        weight_a = abs(d_a1) / (self.radius)
        weight_b = abs(d_b1) / (self.radius)
        weight_o = abs(d_a1 + d_b1) / (2 * self.radius)

        initial_guess = np.array(
            [
                sensor_a[0] * weight_a
                + sensor_b[0] * weight_b
                + sensor_origin[0] * weight_o,
                sensor_a[1] * weight_a
                + sensor_b[1] * weight_b
                + sensor_origin[1] * weight_o,
            ]
        )

        x, y = solve_trilateration(
            sensor_a, sensor_b, sensor_origin, d_a1, d_b1, initial_guess
        )

        return cartesian_to_polar(x, y, self.radius)

    def locate_cc(
        self,
        x: np.ndarray,
        onset_idx: int,
        i: int,
        tol: int = 2,
        left: int = 0,
        right: int = 256,
    ):
        """Locate where an onset was generated.

        :param x: array of shape (N, C) containing sensor values for all C
                  sensors.  Should ideally containg right number of samples
                  before the onset
        :param onset_idx: onset index in [0, N)
        :param i: index (in [0, C)) of sensor where the onset was detected
                  first.  This should be the sensor closest to the onset
                  location.
        :param tol: tolerance used - when potential matches in lateration are
            computed, allows each match to be off by this many locations.
        :param left: number of samples before detected onset to use in
            cross-correlation computation (advised to leave at 0).  Will only
            work if onset_idx - left >= 0
        :param right: number of samples after detected onset to use in
            cross-correlation computation.  Around 5ms of samples are useful.
            If x contains <= this many samples, this setting won't do anything
        """
        self.res[:] = 0
        for j in self.lag_maps[i]:
            lag = find_lag(
                x[onset_idx - left : onset_idx + right, i],
                x[onset_idx - left : onset_idx + right, j],
            )
            self.res += (self.lag_maps[i][j] < lag + tol) & (
                self.lag_maps[i][j] > lag - tol
            )

        # Convert x/y coordinate to polar coordinate
        coord = np.unravel_index(np.argmax(self.res), self.res.shape)
        x = coord[1] - (self.res.shape[1] - 1) / 2
        y = (self.res.shape[0] - 1) / 2 - coord[0]
        return cartesian_to_polar(x, y, self.radius)


def find_lag(a: np.ndarray, b: np.ndarray):
    """Find the lag in number of samples between two audio signals.

    :param a: 1D array containing audio signal
    :param b: 1D array containing audio signal
    """
    cross_corr = np.correlate(a, b, mode="full")
    lag = np.argmax(cross_corr) - (len(a) - 1)
    return lag


def find_lag_multi(a, b, top_n=3):
    """Find n most likely lags in number of samples between two audio signals

    :param a: 1D array containing audio signal
    :param b: 1D array containing audio signal
    :param top_n: Number of likely lags to select
    """
    cross_corr = np.correlate(a, b, mode="full")
    peaks, _ = find_peaks(cross_corr)
    peaks = peaks[np.argsort(-cross_corr[peaks])][:top_n]
    return peaks - len(a) + 1, cross_corr[peaks] ** 2


def lag_map_2d(
    mic_a: tuple[int, int],
    mic_b: tuple[int, int],
    d: int = DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = MEDIUM,
    tol: int = 1,
    c: float | None = None,
):
    """Compute lag map for 2D microphone locations.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param medium: the medium the sound travels through.  One of 'air' or
        'drumhead', the latter for optical/magnetic measurements
    :param tol: lags outside the drum are replaced with np.nan - within some
        tolerance (in centimeters) at the edges.  Note that the
        top/bottom/left/right/edges are naturally at the edge of the matrix,
        tolerance doesn't increase legality there
    :param c: speed of sound. uses speed_of_sound if not provided
    """
    if c is None:
        c = speed_of_sound(100 * scale, medium=medium)

    # This will give us a diameter to use which we can sample at millimeter
    # precision
    r = int(np.round(d * scale / 2))
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    circular_mask = i**2 + j**2 > ((r + tol * scale) ** 2)

    # compute lag in seconds from each potential location to microphones
    lag_a = np.sqrt((i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2) / c
    lag_b = np.sqrt((i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2) / c
    lag_map = np.round((lag_a - lag_b) * sr).astype(np.float32)
    lag_map[circular_mask] = np.nan
    return lag_map


def lag_map_3d(
    mic_a: tuple[int, int, int],
    mic_b: tuple[int, int, int],
    d: int = DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = MEDIUM,
    tol: int = 1,
    c: None | float = None,
):
    """Compute lag map for 3D microphone locations.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param medium: the medium the sound travels through.  One of 'air' or
        'drumhead', the latter for optical/magnetic measurements
    :param tol: lags outside the drum are replaced with np.nan - within some
        tolerance (in centimeters) at the edges.  Note that the
        top/bottom/left/right/edges are naturally at the edge of the matrix,
        tolerance doesn't increase legality there
    :param c: speed of sound. uses speed_of_sound if not provided
    """
    if c is None:
        c = speed_of_sound(100 * scale, medium=medium)
    n = int(np.round(d, 1) * scale)
    r = n // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
    circular_mask = i**2 + j**2 > ((r + tol * scale) ** 2)

    # Z-coordinate of the playing surface
    z_surface = 0

    # compute lag in seconds from each potential location to microphones
    lag_a = (
        np.sqrt(
            (i - mic_a[0]) ** 2
            + (j - mic_a[1]) ** 2
            + (z_surface - mic_a[2]) ** 2
        )
        / c
    )
    lag_b = (
        np.sqrt(
            (i - mic_b[0]) ** 2
            + (j - mic_b[1]) ** 2
            + (z_surface - mic_b[2]) ** 2
        )
        / c
    )

    lag_map = np.round((lag_a - lag_b) * sr).astype(np.float32)
    lag_map[circular_mask] = np.nan
    return lag_map


def sound_intensity_at_source(
    strike_location, strike_force=STRIKE_FORCE, diameter=DIAMETER
) -> float:
    # Placeholder
    return strike_force


def vec_sub(a, b):
    x = a[0] - b[0].reshape(-1)
    y = a[1] - b[1].reshape(-1)
    z = np.full_like(x, a[2] - b[2], dtype=float)
    return np.vstack((x, y, z)).T


def attenuate_intensity(
    source_loc, mic_loc, reflectivity, intensity_at_source
):
    direction_vectors = vec_sub(mic_loc, source_loc)
    distance = np.linalg.norm(direction_vectors, axis=-1)

    # Compute the normal vector to the drumhead
    normal_vector = np.array([0.0, 0.0, 1.0])

    # Normalize the direction vectors
    direction_vectors /= np.linalg.norm(
        direction_vectors, axis=-1, keepdims=True
    )
    # Compute the angle between the direction vector and the normal vector
    thetas = np.arccos(np.dot(direction_vectors, normal_vector))

    # Compute the attenuation factor
    A = (
        intensity_at_source
        * (1 + reflectivity * (1 - np.abs(np.cos(thetas))))
        / (distance)
    )
    return A, np.degrees(thetas)


def lag_intensity_map(
    mic_a: tuple[int, int, int],
    mic_b: tuple[int, int, int],
    reflectivity: float = 0.5,
    d: int = DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = MEDIUM,
):
    """Compute lag and sound intensity maps for 3D microphone locations.

    NOTE: sound intensity dropoff through membrane/drumhead not yet computed
    correctly.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param reflectivity: reflectivity parameter for attenuate_intensity() - the
        larger, the lower the attenuation
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param medium: the medium the sound travels through.  One of 'air' or
        'drumhead', the latter for optical/magnetic measurements
    """
    d = int(np.round(d, 1) * scale)
    r = d // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))

    # Z-coordinate of the playing surface
    z_surface = 0

    # Intensity at the source
    intensity_at_source = 1

    def sound_intensity_at_mic(mic):
        mic_loc = np.array(mic)
        source_loc = (i, j, z_surface)
        A, _ = attenuate_intensity(
            source_loc, mic_loc, reflectivity, intensity_at_source
        )
        return A.reshape(i.shape)

    # Compute lag and signal strength
    lags_a = np.sqrt(
        (i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2 + (z_surface - mic_a[2]) ** 2
    ) / speed_of_sound(100 * scale, medium=medium)
    lags_b = np.sqrt(
        (i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2 + (z_surface - mic_b[2]) ** 2
    ) / speed_of_sound(100 * scale, medium=medium)
    lag_difference = np.round((lags_a - lags_b) * sr)
    signal_strengths_a = 10 * np.log10(sound_intensity_at_mic(mic_a))
    signal_strengths_b = 10 * np.log10(sound_intensity_at_mic(mic_b))

    return (
        lag_difference.astype(np.float32),
        signal_strengths_a.astype(np.float32),
        signal_strengths_b.astype(np.float32),
    )
