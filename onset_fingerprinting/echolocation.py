import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import maximum_filter1d
from scipy.ndimage import binary_opening
import math

# Constants
TEMPERATURE = 20.0  # Celsius
HUMIDITY = 50.0  # percentage
DIAMETER = 14 * 2.54  # centimeters
STRIKE_FORCE = 1.0  # arbitrary units
C = 340.29 * 100


def find_lag(signal1: np.ndarray, signal2: np.ndarray):
    """Find the lag in number of samples between two audio signals.

    :param signal1: 1D array containing audio signal
    :param signal2: 1D array containing audio signal
    """
    cross_corr = np.correlate(signal1, signal2, mode="full")
    lag = np.argmax(cross_corr) - (len(signal1) - 1)
    return lag


def detect_onset_region(
    audio, detected_onset, n=256, median_filter_size=5, threshold_factor=0.5
):
    """In an audio signal around the onset, select the region likely containing
    the onset (removing the relatively quiet part before).

    :param audio: 1D array containing audio signal
    :param detected_onset: index of detected onset inside audio
    :param n: number of samples to look at around the onset
    :param median_filter_size: size of the median filter used before
        thresholding
    :param threshold_factor: threshold the median filtered absolute signal
        above this value to separate loud and quiet parts
    """
    start_idx = max(detected_onset - n // 2, 0)
    end_idx = min(detected_onset + n // 2, len(audio))
    region = audio[start_idx:end_idx]

    # Compute a rolling median filter over the absolute value of the signal
    absolute_region = np.abs(region)
    filtered_signal = medfilt(absolute_region, kernel_size=median_filter_size)

    threshold = threshold_factor * np.max(filtered_signal)
    binary_signal = filtered_signal > threshold

    # Apply binary morphology and find the first True index
    binary_signal_morphed = binary_opening(binary_signal, structure=np.ones(5))
    onset_idx_in_region = np.argmax(binary_signal_morphed)

    # Translate back to the original audio array index
    onset_idx_in_audio = start_idx + onset_idx_in_region

    return onset_idx_in_audio



def sim_lag_centered(
    mic_a: tuple[int, int], mic_b: tuple[int, int], d=DIAMETER, sr=96000
):
    # This will give us a diameter to use which we can sample at millimeter
    # precision
    n = int(np.round(d, 1) * 10)
    r = n // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))

    # compute lag in seconds from each potential location to microphones
    lag_a = np.sqrt((i - mic_a[0]) ** 2 + (j - mic_a[1]) ** 2) / C
    lag_b = np.sqrt((i - mic_b[0]) ** 2 + (j - mic_b[1]) ** 2) / C
    return np.round((lag_a - lag_b) * sr)


def sim_lag_3d(
    mic_a: tuple[int, int, int],
    mic_b: tuple[int, int, int],
    d=DIAMETER,
    sr=96000,
):
    n = int(np.round(d, 1) * 10)  # Diameter in millimeters
    r = n // 2
    i, j = np.meshgrid(range(-r, r + 1), range(-r, r + 1))

    # Z-coordinate of the playing surface
    z_surface = 0

    # compute lag in seconds from each potential location to microphones
    lag_a = (
        np.sqrt(
            (i - mic_a[0]) ** 2
            + (j - mic_a[1]) ** 2
            + (z_surface - mic_a[2]) ** 2
        )
        / C
    )
    lag_b = (
        np.sqrt(
            (i - mic_b[0]) ** 2
            + (j - mic_b[1]) ** 2
            + (z_surface - mic_b[2]) ** 2
        )
        / C
    )

    return np.round((lag_a - lag_b) * sr)


def polar_to_cartesian(r, theta):
    theta_radians = math.radians(theta) - np.pi / 2
    x = r * math.cos(theta_radians)
    y = r * math.sin(theta_radians)
    return x, y


def cartesian_to_polar(x, y, r=None):
    r = np.sqrt(x**2 + y**2)
    theta_radians = np.arctan2(y, x) + np.pi / 2

    # Adjust theta to be in the range [0, 2 * pi)
    theta_radians = theta_radians % (2 * np.pi)

    return r, theta_radians * 180 / np.pi


def spherical_to_cartesian(r, theta, phi):
    theta_radians = np.radians(theta)
    phi_radians = np.radians(phi)

    x = r * np.sin(theta_radians) * np.cos(phi_radians)
    y = r * np.sin(theta_radians) * np.sin(phi_radians)
    z = r * np.cos(theta_radians)

    return x, y, z


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta_radians = np.arccos(z / r)
    phi_radians = np.arctan2(y, x)

    # Adjust phi to be in the range [0, 2 * pi)
    phi_radians = phi_radians % (2 * np.pi)

    return r, np.degrees(theta_radians), np.degrees(phi_radians)


def speed_of_sound(temperature=TEMPERATURE, humidity=HUMIDITY) -> float:
    return 331.5 + 0.6 * temperature + 0.012 * humidity


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
    # Vectorized version of the above
    # direction_vectors = mic - sources
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


def sim_3d(
    mic_a: tuple[int, int, int],
    mic_b: tuple[int, int, int],
    reflectivity=0.5,
    d=14 * 2.54,
    sr=96000,
):
    # Diameter in millimeters
    d = int(np.round(d, 1) * 10)
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
    lags_a = (
        np.sqrt(
            (i - mic_a[0]) ** 2
            + (j - mic_a[1]) ** 2
            + (z_surface - mic_a[2]) ** 2
        )
        / C
    )
    lags_b = (
        np.sqrt(
            (i - mic_b[0]) ** 2
            + (j - mic_b[1]) ** 2
            + (z_surface - mic_b[2]) ** 2
        )
        / C
    )
    lag_difference = np.round((lags_a - lags_b) * sr)
    signal_strengths_a = 10 * np.log10(sound_intensity_at_mic(mic_a))
    signal_strengths_b = 10 * np.log10(sound_intensity_at_mic(mic_b))
    # signal_strengths_a = sound_intensity_at_mic(mic_a)
    # signal_strengths_b = sound_intensity_at_mic(mic_b)

    return lag_difference, signal_strengths_a, signal_strengths_b
