import ctypes
from pathlib import Path
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


class ButterworthFilter:
    """
    Butterworth filter to apply to multiple signals in parallel.
    """

    def __init__(self, cutoff, n, order=2, sr=44100, btype="high"):
        self.b, self.a = sig.butter(
            order, cutoff, btype=btype, analog=False, output="ba", fs=sr
        )
        self.b, self.a = np.float32(self.b), np.float32(self.a)
        self.zi = np.zeros((order, n), dtype=np.float32)

    def __call__(self, x: np.ndarray):
        y, self.zi = sig.lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y


class AREnvelopeFollower:
    """
    Attack-Release envelope follower which follows multiple envelopes at the
    same time.  For use in parallel onset detection of several signals.

    Need to install ARenvelope.c as a shared DLL in the same directory as this
    file, for example: gcc -shared -o ARenvelope.so -fPIC -Ofast ARenvelope.c
    """

    def __init__(self, x0: np.ndarray, attack=3, release=383):
        self.attack = np.float32(1 / attack)
        self.release = np.float32(1 / release)
        self.y = x0
        self.c_ar_env = ctypes.CDLL(Path(__file__) / "ARenvelope.so")
        self.c_ar_env.process.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
            ),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.n, self.size = np.int32(x0.shape)

    def __call__(self, x):
        self.c_ar_env.process(
            x, self.y, self.attack, self.release, self.size, self.n
        )
        return self.y


class AmplitudeOnsetDetector:
    """
    Multi-channel amplitude/time-domain onset detector ideal for very fast
    detection of percussive onsets.

    Compares two (fast & slow) amplitude envelope followers to catch large
    relative amplitude spikes as onsets.

    Mostly matches FluCoMa's AmpSlice object:
    https://learn.flucoma.org/reference/ampslice/
    """

    def __init__(
        self,
        n_signals: int,
        block_size: int = 32,
        floor: float = -70.0,
        hipass_freq: float = 2000.0,
        fast_ar: tuple[float, float] = (3.0, 383.0),
        slow_ar: tuple[float, float] = (2205.0, 2205.0),
        on_threshold: float = 19.0,
        off_threshold: float = 8.0,
        cooldown: int = 1323,
        sr: int = 44100,
    ):
        """
        :param n_signals: number of signals to individually detect onsets on.
            Arrays used to detect on need to be of shape [block_size,
            n_signals]
        :param block_size: number of samples processed at once / buffer size
        :param floor: threshold in dB below which no onsets are detected
        :param hipass_freq: if not 0, applies 4th order Butterworth highpass
            filter to the signal before
        :param fast_ar: tuple of attack/release values for fast envelope
            follower.  Uses the reciprocal (1/x) of these values as the
            exponential smoothing parameter for increases/decreases in
            amplitude.  For example, an attack of 3 will mean that the EWMA
            will move towards new samples at a rate of 1/3.

            Detection speed is mostly controlled by how low the fast attack
            (fast_ar[0]) is, as this controls how quickly on_threshold can be
            reached
        :param slow_ar: tuple of attack/release values for slow envelope
            follower.  Uses the reciprocal (1/x) of these values as the
            exponential smoothing parameter for increases/decreases in
            amplitude
        :param on_threshold: threshold in dB above which an onset is detected
        :param off_threshold: threshold in dB below which the signal must fall
            before another onset can be detected
        :param cooldown: after an onset has been detected, will wait at least
            this many samples before triggering another onset, regardless of
            on/off thresholds
        :param sr: sample rate
        """
        self.n_signals = n_signals
        self.block_size = block_size
        self.floor = floor
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.cooldown = cooldown
        self.sr = sr

        self.hp = ButterworthFilter(hipass_freq, n_signals, 4, sr, "high")
        self.fast_slide = AREnvelopeFollower(
            np.full((block_size, n_signals), floor, dtype=np.float32), *fast_ar
        )
        self.slow_slide = AREnvelopeFollower(
            np.full((block_size, n_signals), floor, dtype=np.float32), *slow_ar
        )

        self.state = np.zeros(n_signals, dtype=bool)
        self.prev_values = np.zeros(n_signals)
        self.debounce_count = np.zeros(n_signals, dtype=int)

    def __call__(self, x: np.ndarray) -> tuple[list[int], list[int]]:
        """
        Detect onsets for new samples.

        :param x: array of size [block_size, n_signals]

        :returns: tuple of [channels, deltas], where channels is a list of
                  channel indexes in [0, n_signals), and deltas is a list of
                  the same size as channels containing onset deltas wrt.  the
                  first sample in x.

            For example, ([1], [4]) indicates an offset in channel 1, detected
            at the 4th sample in x (this onset is not backtracked, so the true
            onset lies some samples earlier, depending mostly on the fast
            attack parameter)
        """
        if self.hp is not None:
            samples = self.hp(x)
        # Compute floor-clipped, rectified dB
        samples = np.maximum(20 * np.log10(np.abs(samples)), self.floor)
        values = self.fast_slide(x) - self.slow_slide(x)

        # Logic for detection
        crossed_on_threshold = (
            (values > self.on_threshold)
            & (~self.state)
            & (self.debounce_count < 1)
        )
        crossed_on_threshold[0] &= self.prev_values < self.on_threshold
        crossed_on_threshold[1:] &= values[:-1] < self.on_threshold

        # Find the first index where the on_threshold is crossed & adjust for
        # first row
        on_indices = np.argmax(crossed_on_threshold, axis=0)
        on = (on_indices > 0) | crossed_on_threshold[0, :]

        # Update debounce_count and state for channels that detected an onset
        self.state[on] = True
        self.debounce_count[on] = self.cooldown
        self.debounce_count[self.debounce_count > 0] -= self.block_size
        # This would be more accurate but is probably not necessary
        # deb = np.array([block_size] * n_signals)
        # deb[on] -= on_indices[on]
        # debounce_count -= deb

        # Update states for off_threshold crossing
        # only check for off_threshold after detection to turn off
        crossed_off_threshold = values < self.off_threshold

        # Again, this is more accurate, but it's quicker to use the max
        # for i, ind in enumerate(on_indices):
        #     crossed_off_threshold[:ind, i] = False
        crossed_off_threshold[: on_indices.max(), :] = False
        self.state[np.any(crossed_off_threshold, axis=0)] = False
        self.prev_values[:] = values[-1, :]

        # if any(on_indices > 0):
        #     print(np.where(on)[0], on_indices[on])

        # Channels and deltas
        return on_indices[on], np.where(on)[0]


def setup(audio, block_size=32):
    floor = params["floor"]
    hi_pass_freq = params["hi_pass_freq"]
    sample_rate = params["sample_rate"]
    hi_pass_freq = min(hi_pass_freq / sample_rate, 0.5)
    sample_rate = None  # otherwise we get wrong hpf if switching
    fast_ramp_up = params["fast_ramp_up"]
    slow_ramp_up = params["slow_ramp_up"]
    fast_ramp_down = params["fast_ramp_down"]
    slow_ramp_down = params["slow_ramp_down"]
    on_threshold = params["on_threshold"]
    off_threshold = params["off_threshold"]
    debounce = params["debounce"]

    n_signals = audio.shape[1]
    hp1 = ButterworthFilter(hi_pass_freq, n_signals, 4, sample_rate, "high")
    fast_slide = AREnvelopeFollower(
        np.full((block_size, n_signals), floor, dtype=np.float32),
        fast_ramp_up,
        fast_ramp_down,
    )
    slow_slide = AREnvelopeFollower(
        np.full((block_size, n_signals), floor, dtype=np.float32),
        slow_ramp_up,
        slow_ramp_down,
    )

    state = np.zeros(n_signals, dtype=bool)
    prev_values = np.zeros(n_signals)
    debounce_count = np.zeros(n_signals, dtype=int)
    num_samples, n_signals = audio.shape
    output = np.zeros((num_samples, n_signals), dtype=int)

    num_blocks = len(audio) // block_size
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size
        samples = audio[start_idx:end_idx]

        values = pe(samples, floor, fast_slide, slow_slide, hp1, hp2)

        # Logic for detection
        crossed_on_threshold = (
            (values > on_threshold) & (~state) & (debounce_count < 1)
        )
        crossed_on_threshold[0] &= prev_values < on_threshold
        crossed_on_threshold[1:] &= values[:-1] < on_threshold

        # Find the first index where the on_threshold is crossed & adjust for
        # first row
        on_indices = np.argmax(crossed_on_threshold, axis=0)
        on = (on_indices > 0) | crossed_on_threshold[0, :]

        # Update debounce_count and state for channels that detected an onset
        state[on] = True
        debounce_count[on] = debounce
        debounce_count[debounce_count > 0] -= block_size
        # This would be more accurate but is probably not necessary
        # deb = np.array([block_size] * n_signals)
        # deb[on] -= on_indices[on]
        # debounce_count -= deb

        # if any(on_indices > 0):
        #     print(np.where(on)[0], on_indices[on])
        output[start_idx:end_idx][on_indices[on], np.where(on)[0]] = 1

        # Update states for off_threshold crossing
        # only check for off_threshold after detection to turn off
        crossed_off_threshold = values < off_threshold

        # Again, this is more accurate, but it's quicker to use the max
        # for i, ind in enumerate(on_indices):
        #     crossed_off_threshold[:ind, i] = False
        crossed_off_threshold[: on_indices.max(), :] = False
        state[np.any(crossed_off_threshold, axis=0)] = False
        prev_values[:] = values[-1, :]

    return output


def pe(samples, floor, fast_slide, slow_slide, hp=None):
    if hp is not None:
        samples = hp(samples)
    # Compute floor-clipped, rectified dB
    samples = np.maximum(20 * np.log10(np.abs(samples)), floor)
    return fast_slide(samples) - slow_slide(samples)
