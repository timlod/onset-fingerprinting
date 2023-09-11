import numpy as np
import librosa
from scipy import signal as sig
from scipy.ndimage import binary_opening


def detect_onsets(
    x: np.ndarray,
    n_fft: int = 256,
    hop: int = 32,
    sr: int = 96000,
    return_oe: bool = False,
):
    D = np.abs(librosa.stft(x, hop_length=hop, n_fft=n_fft))
    freq = np.fft.fftfreq(n_fft, 1 / sr)[: len(D)]
    # Use only high frequency spectral flux to better catch low-energy hits
    # possibly use a log-weight instead of just cutting
    # lowcut = 12000
    # lowcut_idx = np.where(freq > lowcut)[0][0]
    # oe = D[lowcut_idx:, 1:] - D[lowcut_idx:, :-1]

    # Weight frequencies to de-emphasise low frequency content
    aw = librosa.A_weighting(freq)[:, None]
    D *= (aw - aw.min()) / np.abs(aw.min())

    oe = D[:, 1:] - D[:, :-1]
    oe = np.maximum(0.0, oe)
    oe = oe.mean(0)
    oe /= np.percentile(oe, 99.9)

    peaks = librosa.util.peak_pick(
        oe,
        pre_max=0.12 * sr // hop,
        post_max=0.01 * sr // hop,
        pre_avg=0.12 * sr // hop,
        post_avg=0.01 * sr // hop + 1,
        delta=0.1,
        wait=sr * 0.07 // hop,
    )
    # - hop is more of a backtracking operation - most likely the peak will be
    # - right before the peak
    peaks = peaks * hop  # - hop
    if return_oe:
        return peaks, oe
    else:
        return peaks


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
    filtered_signal = sig.medfilt(
        absolute_region, kernel_size=median_filter_size
    )

    threshold = threshold_factor * np.max(filtered_signal)
    binary_signal = filtered_signal > threshold

    # Apply binary morphology and find the first True index
    binary_signal = binary_opening(binary_signal, structure=np.ones(5))
    onset_idx = np.argmax(binary_signal)
    return start_idx + onset_idx
