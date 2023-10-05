from loopmate.circular_array import CircularArray
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
        self.c_ar_env = ctypes.CDLL(Path(__file__).parent / "ARenvelope.so")
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

    Example usage::


        block_size = 32
        od = AmplitudeOnsetDetector(4)
        num_blocks = len(audio) // block_size
        channels, onsets = [], []
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            samples = audio[start_idx:end_idx]
            c, d = od(samples)
            if len(c) > 0:
                channels.append(c)
                onsets.append([start_idx + x for x in d])
        channels = [x for y in channels for x in y]
        onsets = [x for y in onsets for x in y]
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
        backtrack: bool = False,
        backtrack_buffer_size: int = 80,
        backtrack_smooth_size: int = 7,
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

        self.hp = (
            ButterworthFilter(hipass_freq, n_signals, 4, sr, "high")
            if hipass_freq != 0
            else None
        )
        self.fast_slide = AREnvelopeFollower(
            np.full((block_size, n_signals), floor, dtype=np.float32), *fast_ar
        )
        self.slow_slide = AREnvelopeFollower(
            np.full((block_size, n_signals), floor, dtype=np.float32), *slow_ar
        )

        self.state = np.zeros(n_signals, dtype=bool)
        self.prev_values = np.zeros(n_signals)
        self.debounce_count = np.zeros(n_signals, dtype=int)

        self.backtrack = backtrack
        if backtrack:
            assert (
                block_size <= backtrack_buffer_size
            ), "backtrack_buffer_size should be at least block_size!"
            self.buffer = CircularArray(
                np.empty((backtrack_buffer_size, n_signals), dtype=np.float32)
            )
            self.smoother = np.ones(
                (backtrack_smooth_size, 1), dtype=np.float32
            )

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
            x = self.hp(x)
        # Compute floor-clipped, rectified dB
        x = np.maximum(20 * np.log10(np.abs(x)), self.floor)
        relative_envelope = self.fast_slide(x) - self.slow_slide(x)
        if self.backtrack:
            self.buffer.write(relative_envelope)

        # Logic for detection
        crossed_on_threshold = (
            (relative_envelope > self.on_threshold)
            & (~self.state)
            & (self.debounce_count < 1)
        )
        crossed_on_threshold[0] &= self.prev_values < self.on_threshold
        crossed_on_threshold[1:] &= relative_envelope[:-1] < self.on_threshold

        # Find the first index where the on_threshold is crossed & adjust for
        # first row
        on_indices = np.argmax(crossed_on_threshold, axis=0)
        on = (on_indices > 0) | crossed_on_threshold[0, :]

        # Update debounce_count and state for channels that detected an onset
        self.state[on] = True
        self.debounce_count[on] = self.cooldown
        self.debounce_count[self.debounce_count > 0] -= self.block_size

        # Update states for off_threshold crossing
        # only check for off_threshold after detection to turn off
        crossed_off_threshold = relative_envelope < self.off_threshold

        crossed_off_threshold[: on_indices.max(), :] = False
        self.state[np.any(crossed_off_threshold, axis=0)] = False
        self.prev_values[:] = relative_envelope[-1, :]

        # Channels and deltas
        channels, deltas = np.where(on)[0], on_indices[on]
        if self.backtrack and len(channels) > 0:
            deltas = self.backtrack_onsets(channels, deltas)
        return channels, deltas, relative_envelope

    def backtrack_onsets(self, channels, deltas):
        buffer = self.buffer[-self.buffer.N :]
        # Do some smoothing to allow to 'roll over' plateaus
        if len(self.smoother) > 1:
            buffer = sig.convolve(buffer, self.smoother)
        new_deltas = []
        for channel, delta in zip(channels, deltas):
            i = self.block_size - delta
            prev = buffer[-i - 1, channel]
            current = buffer[-i, channel]
            while (i + 1 < self.buffer.N) and (current > prev):
                delta -= 1
                i += 1
                current = prev
                prev = buffer[-i - 1, channel]
            new_deltas.append(delta)
        return new_deltas
