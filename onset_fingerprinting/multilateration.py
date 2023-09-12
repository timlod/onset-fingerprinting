import numpy as np
from scipy.signal import find_peaks
from onset_fingerprinting import detection
import math

TEMPERATURE = 20.0
HUMIDITY = 50.0
DIAMETER = 14 * 2.54
STRIKE_FORCE = 1.0
# speed in m/s of sound through drumhead membrane
C_drumhead = 82.0
# medium used in sound propagation equations (air or drumhead)
MEDIUM = "air"


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
        return scale * 331.5 + 0.6 * temperature + 0.012 * humidity
    else:
        return scale * C_drumhead


def polar_to_cartesian(r: float, theta: float):
    """Convert 2D polar coordinates to cartesian coordinates.

    :param r: radius
    :param theta: angle in degrees
    """
    theta_radians = math.radians(theta) - np.pi / 2
    x = r * math.cos(theta_radians)
    y = r * math.sin(theta_radians)
    return x, y


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

    theta_radians = np.arctan2(y, x) + np.pi / 2

    # Adjust theta to be in the range [0, 2 * pi)
    theta_radians = theta_radians % (2 * np.pi)

    return r, theta_radians * 180 / np.pi


def spherical_to_cartesian(r: float, theta: float, phi: float):
    """Convert 3D spherical/polar coordinates to cartesian coordinates

    :param r: radius
    :param theta: vertical angle (along z-axis) in degrees
    :param phi: horizontal angle (along x-axis) in degrees
    """
    theta_radians = np.radians(theta)
    phi_radians = np.radians(phi)

    x = r * np.sin(theta_radians) * np.cos(phi_radians)
    y = r * np.sin(theta_radians) * np.sin(phi_radians)
    z = r * np.cos(theta_radians)

    return x, y, z


def cartesian_to_spherical(x: float, y: float, z: float):
    """Convert 3D cartesian coordinates to spherical/polar coordinates.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta_radians = np.arccos(z / r)
    phi_radians = np.arctan2(y, x)

    # Adjust phi to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)

    return r, np.degrees(theta_radians), np.degrees(phi_radians)


class Multilaterate:
    # TODO: if predicted location falls outside of circle, reject (includes [0,
    # 0], where failures would end up)

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

        self.lag_maps = [{} for _ in range(len(self.sensor_locs))]
        for i in range(len(self.sensor_locs)):
            for k in [-1, 1]:
                # Wrap around to first/last mic_loc
                j = (i + k) % len(self.sensor_locs)
                self.lag_maps[i][j] = echolocation.lag_map_2d(
                    self.sensor_locs[i],
                    self.sensor_locs[j],
                    d=d,
                    sr=sr,
                    scale=scale,
                    medium="drumhead",
                )
        # Pre-allocating results array - possibly useless optimization
        self.res = np.zeros_like(self.lag_maps[0][1])

    def locate(
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
        onset_idx = detection.detect_onset_region(
            x[:, i], onset_idx, right, threshold_factor=0.2
        )
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

    def locate_given_lags(self, lags: list[int], i: int, tol: int = 2):
        """Locate where an onset was generated given lags between sensors
        computed elsewhere.

        :param lags: lags between sensor closest to onset (should be the sensor
            on whose data the onset was first detected) and the two closest
            sensors, left-to-right.  For example, if the 'north' sensor
            triggers an onset, this should contain a list of lags for the
            'east' and 'west' sensors, in that order
        :param i: index (in [0, C)) of sensor where the onset was detected
                  first.  This should be the sensor closest to the onset
                  location.
        :param tol: grid tolerance - when potential matches in lateration are
            computed, allows each match to be off by this many locations.
            Default of 2 means that for millimeter resolution, a 5mm region
            around the potential onset locations are considered.  Higher
            tolerance
        """
        self.res[:] = 0
        for j, lag in zip(self.lag_maps[i], lags):
            self.res += (self.lag_maps[i][j] < lag + tol) & (
                self.lag_maps[i][j] > lag - tol
            )
        # Convert x/y coordinate to polar coordinate
        coord = np.unravel_index(np.argmax(self.res), self.res.shape)
        x = coord[1] - (self.res.shape[1] - 1) / 2
        y = (self.res.shape[0] - 1) / 2 - coord[0]
        return cartesian_to_polar(x, y, self.radius)

    def locate_given_lags_accurate(
        self, lags: list[int], i: int, tol: int = 2
    ):
        """Locate where an onset was generated given lags between sensors
        computed elsewhere.  Instead of taking the max, it indexes the centroid
        of the blob that matches.  Returns None if no match, as opposed to [0,
        0] for the other methods.

        :param lags: lags between sensor closest to onset (should be the sensor
            on whose data the onset was first detected) and the two closest
            sensors, left-to-right.  For example, if the 'north' sensor
            triggers an onset, this should contain a list of lags for the
            'east' and 'west' sensors, in that order
        :param i: index (in [0, C)) of sensor where the onset was detected
                  first.  This should be the sensor closest to the onset
                  location.
        :param tol: grid tolerance - when potential matches in lateration are
            computed, allows each match to be off by this many locations.
            Default of 2 means that for millimeter resolution, a 5mm region
            around the potential onset locations are considered.  Higher
            tolerance
        """
        js = list(self.lag_maps[i].keys())
        res = (
            (self.lag_maps[i][js[0]] < (lags[0] + tol))
            & (self.lag_maps[i][js[0]] > (lags[0] - tol))
            & (self.lag_maps[i][js[1]] < (lags[1] + tol))
            & (self.lag_maps[i][js[1]] > (lags[1] - tol))
        )
        row, col = np.where(res)
        if len(row) == 0:
            return None
        # Had to swap them here, haven't taken time to check why
        x, y = col.mean(), row.mean()
        x -= (self.res.shape[1] - 1) / 2
        y = (self.res.shape[0] - 1) / 2 - y
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
    """
    # This will give us a diameter to use which we can sample at millimeter
    # precision
    n = int(np.round(d, 1) * scale)
    r = n // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))

    # compute lag in seconds from each potential location to microphones
    lag_a = np.sqrt((i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2) / (
        speed_of_sound(100 * scale, medium=medium)
    )
    lag_b = np.sqrt((i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2) / (
        speed_of_sound(100 * scale, medium=medium)
    )
    return np.round((lag_a - lag_b) * sr).astype(np.float32)


def lag_map_3d(
    mic_a: tuple[int, int, int],
    mic_b: tuple[int, int, int],
    d: int = DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = MEDIUM,
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
    """

    n = int(np.round(d, 1) * scale)
    r = n // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))

    # Z-coordinate of the playing surface
    z_surface = 0

    # compute lag in seconds from each potential location to microphones
    lag_a = np.sqrt(
        (i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2 + (z_surface - mic_a[2]) ** 2
    ) / speed_of_sound(100 * scale, medium=medium)
    lag_b = np.sqrt(
        (i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2 + (z_surface - mic_b[2]) ** 2
    ) / speed_of_sound(100 * scale, medium=medium)

    return np.round((lag_a - lag_b) * sr).astype(np.float32)


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
