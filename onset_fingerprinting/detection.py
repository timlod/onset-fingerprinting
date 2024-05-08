import ctypes
from pathlib import Path

import librosa
import numpy as np
from loopmate.circular_array import CircularArray
from scipy import signal as sig
from scipy.ndimage import binary_opening, maximum_filter1d


def detect_onsets(x: np.ndarray, sr: int = 96000, method="amp"):
    if method == "amp":
        return detect_onsets_amplitude(x, sr=sr)
    else:
        return detect_onsets_spectral(x, sr=sr)


def detect_onsets_amplitude(
    x: np.ndarray,
    block_size: int = 128,
    floor: float = -70.0,
    hipass_freq: float = 2000.0,
    fast_ar: tuple[float, float] = (3.0, 383.0),
    slow_ar: tuple[float, float] = (2205.0, 2205.0),
    on_threshold: float = 0.5,
    off_threshold: float = 0.1,
    cooldown: int = 1323,
    backtrack: bool = False,
    backtrack_buffer_size: int = 80,
    backtrack_smooth_size: int = 5,
    sr: int = 96000,
):
    od = AmplitudeOnsetDetector(
        x.shape[1],
        block_size,
        hipass_freq=2000,
        fast_ar=(1, 1000),
        slow_ar=(10000, 10000),
        on_threshold=0.6,
        off_threshold=0.01,
        cooldown=1323,
        sr=sr,
        backtrack=True,
        backtrack_buffer_size=2 * block_size,
        backtrack_smooth_size=1,
    )
    od.init_minmax_tracker(x[: int(0.5 * sr)])
    channels, onsets = [], []
    rel = []
    for i in range(0, len(x), block_size):
        samples = x[i : i + block_size]
        c, d, r = od(samples)
        rel.append(r)
        if len(c) > 0:
            d = [i + x for x in d]
            channels.append(c)
            onsets.append(d)
    channels_flat = [x for y in channels for x in y]
    onsets_flat = [x for y in onsets for x in y]
    return channels_flat, onsets_flat


def detect_onsets_spectral(
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


def cross_correlation_lag(
    x: np.ndarray,
    y: np.ndarray,
    legal_lags: tuple[int, int],
    d: int = 0,
    normalization_cutoff: int = 10,
) -> int:
    """
    Compute cross-correlation (CC) of two sequences, normalizes the resulting
    lags by contribution (i.e. divides each value by the number of elements
    which contributed to that lag) and chooses the maximum within a given legal
    region.  Also allows to compute CC on nth-order differences/derivatives.

    The cross correlation is the dot product of parts of the two sequences.
    This includes over the amount of elements present for a given lag.  Since
    only the center value (lag 0) has contributions from all elements, it has
    the potential to be the largest value (see np.correlate(np.ones(5),
    np.ones(5), mode="full"), with the result of 1:5:1).  If we want to weigh
    smaller matching subsequences equally, we need do divide each lag by the
    number of elements which contributed to it (the aforementioned example
    would give a cross-correlation for 1 at each lag in that case).

    :param x: first input audio
    :param y: second input audio
    :param legal_lags: which lags to consider in the cross-correlation.  Use to
        incorporate knowledge of sensor placement into choosing the correct lag
    :param d: computes the CC on the d-th difference/derivative
    :param normalization_cutoff: number of elements which need to be present in
        one lag of the CC to be normalized such that that lag can contribute
        equally to lag as other lags above cutoff.  See description for better
        explanation
    """
    x = np.diff(x, d)
    y = np.diff(y, d)
    n = len(x)
    cc = np.correlate(x, y, "full")
    # Normalize such that each cc value with contributions of more than cutoff
    # amount of elements has an equal chance to be the top lag
    normalizer = np.arange(len(x)) + 1
    normalizer[:normalization_cutoff] = normalization_cutoff
    cc[:n] /= normalizer
    cc[n:] /= normalizer[n - 2 :: -1]
    # look at only legal lags - here we always assume that x is before y, and
    # legality means the lag from x to y (usually positive) - these lags need
    # to be negated to fit cc (where lags before n represent y needing to move
    # forward)
    # TODO: add tolerance zone to legality? : perhaps better to add before call
    cc = cc[n - legal_lags[1] : n - legal_lags[0]]
    return -(np.argmax(cc) - legal_lags[1])


def adjust_onset(
    onsets: list[int, int], relx: np.ndarray, rely: np.ndarray, new_lag: int
) -> tuple[int, int]:
    """Adjust one onset in a pair based on a target lag and the signals'
    respective relative envelopes.

    :param onsets: [onset_x, onset_y]
    :param relx: relative envelope of x
    :param rely: relative envelope of y
    :param new_lag: target lag (likely output of cross_correlation_lag)
    """
    oa = onsets[0]
    ob = onsets[1]
    lag = ob - oa
    lag_diff = lag - new_lag
    # if lag_diff < 0 we need to look before onsets[0] or after onsets[1] and
    # vice versa
    # these will be positive if the new location is larger than the old one
    # we'll take the one with a larger positive change
    da = relx[oa + lag_diff] - relx[oa]
    db = rely[ob - lag_diff] - rely[ob]
    if da > db:
        oa += lag_diff
    else:
        ob -= lag_diff
    return oa, ob


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
        self.c_ar_env = ctypes.CDLL(
            Path(__file__).parent / "envelope_follower.so"
        )
        self.c_ar_env.ar_envelope.argtypes = [
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
        self.c_ar_env.ar_envelope(
            x, self.y, self.attack, self.release, self.size, self.n
        )
        return self.y


class MinMaxEnvelopeFollower:
    """EMA min/max tracker for multi-channel signals.

    This class leverages a C shared library for performance.

    Install the shared DLL by compiling EMA_MinMaxTracker.c, for example:
    gcc -shared -o EMA_MinMaxTracker.so -fPIC -Ofast EMA_MinMaxTracker.c
    """

    def __init__(
        self, x0: np.ndarray, alpha_min=1e-5, alpha_max=1e-5, minmin=0.0
    ):
        self.alpha_min = np.float32(alpha_min)
        self.alpha_max = np.float32(alpha_max)
        self.minmin = np.float32(minmin)
        self.min_val = np.float32(np.min(x0, axis=0))
        self.max_val = np.float32(np.max(x0, axis=0))

        self.c_tracker = ctypes.CDLL(
            Path(__file__).parent / "envelope_follower.so"
        )

        self.c_tracker.minmax_envelope.argtypes = [
            np.ctypeslib.ndpointer(
                dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
            ),
            np.ctypeslib.ndpointer(
                dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"
            ),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.n_samples, self.n_channels = np.int32(x0.shape)

    def __call__(self, x):
        self.c_tracker.minmax_envelope(
            x,
            self.min_val,
            self.max_val,
            self.alpha_min,
            self.alpha_max,
            self.minmin,
            len(x),
            self.n_channels,
        )
        return self.min_val, self.max_val


class AmplitudeOnsetDetector:
    """
    Multi-channel amplitude/time-domain onset detector ideal for very fast
    detection of percussive onsets.

    Compares two (fast & slow) amplitude envelope followers to catch large
    relative amplitude spikes as onsets.

    Originally based on FluCoMa's AmpSlice object:
    https://learn.flucoma.org/reference/ampslice/

    Major changes are block-wise computation of multiple signals at once,
    addition of backtracking (different to how it's done in FluCoMa's AmpGate
    object), and thresholds relative to the min/max envelope of the relative
    envelope as opposed to absolute dB thresholds defined on the relative
    envelope.  The latter is done as different signals and gain settings would
    lead to different thresholds required (for each signal/channel).

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
        on_threshold: float = 0.5,
        off_threshold: float = 0.1,
        cooldown: int = 1323,
        backtrack: bool = False,
        backtrack_buffer_size: int = 80,
        backtrack_smooth_size: int = 5,
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
        :param on_threshold: threshold in [0, 1] above which an onset is
            detected.  Uses the recent range of the relative envelope to
            compute actual thresholds
        :param off_threshold: threshold in [0, 1] below which the signal must
            fall before another onset can be detected.  Uses the recent range
            of the relative envelope to compute actual thresholds
        :param cooldown: after an onset has been detected, will wait at least
            this many samples before triggering another onset, regardless of
            on/off thresholds
        :param backtrack: if True, backtracks each onset to the likely start
        :param backtrack_buffer_size: size of the buffer to keep for
            backtracking onsets. Should be at least equal to block_size
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
        self.minmax_tracker = MinMaxEnvelopeFollower(
            x0=np.array([[0, 10]] * n_signals).T,
            alpha_min=1e-4,
            alpha_max=1e-5,
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
            self.b_alpha = np.float32(2 / (backtrack_smooth_size + 1))
            self.b_tol = np.float32(
                (1 - self.b_alpha) ** backtrack_buffer_size
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
        x = 20 * np.log10(np.abs(x + 1e-10))
        relative_envelope = self.fast_slide(x) - self.slow_slide(x)
        if self.backtrack:
            self.buffer.write(relative_envelope)

        mi, ma = self.minmax_tracker(relative_envelope)
        # Logic for detection
        # on_threshold = self.on_threshold
        on_threshold = ma * self.on_threshold + mi
        crossed_on_threshold = (
            (relative_envelope > on_threshold)
            & (~self.state)
            & (self.debounce_count < 1)
        )
        crossed_on_threshold[0] &= self.prev_values < on_threshold
        crossed_on_threshold[1:] &= relative_envelope[:-1] < on_threshold

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
        # off_threshold = self.off_threshold
        off_threshold = ma * self.off_threshold + mi
        crossed_off_threshold = relative_envelope < off_threshold

        crossed_off_threshold[: on_indices.max(), :] = False
        self.state[np.any(crossed_off_threshold, axis=0)] = False
        self.prev_values[:] = relative_envelope[-1, :]

        # Channels and deltas
        channels, deltas = np.where(on)[0], on_indices[on]
        if self.backtrack and len(channels) > 0:
            deltas = self.backtrack_onsets(channels, deltas)
        return channels, deltas, relative_envelope

    def backtrack_onsets(self, channels, deltas):
        N = self.buffer.N
        buffer = self.buffer[-N:]
        alpha = self.b_alpha
        omba = np.float32(1 - self.b_alpha)
        tol = self.b_tol

        for j in range(len(channels)):
            channel, delta = channels[j], deltas[j]
            i = self.block_size - delta
            current_smoothed = buffer[-i, channel]
            i += 1
            prev = buffer[-i, channel]
            prev_smoothed = alpha * prev + omba * current_smoothed
            while (
                ((current_smoothed > prev_smoothed))
                and (abs(prev_smoothed - prev) > tol)
                and (i + 1 < N)
            ):
                deltas[j] -= 1
                i += 1
                current_smoothed = prev_smoothed
                prev = buffer[-i, channel]
                prev_smoothed = alpha * prev + omba * current_smoothed
        return deltas

    def init_minmax_tracker(self, x):
        if self.hp is not None:
            x = self.hp(x)
        # Compute floor-clipped, rectified dB
        x = 20 * np.log10(np.abs(x + 1e-10))
        for i in range(0, len(x), self.block_size):
            xi = x[i : i + self.block_size, :]
            self.minmax_tracker(self.fast_slide(xi) - self.slow_slide(xi))

    def init(self, x):
        """Initialize onset detector with a data containing stretches of
        silence as well as stretches of audio approximately as loud as it will
        get during performance.

        :param x: multi-channel audio array
        """

        if self.hp is not None:
            x = self.hp(x)
        # Compute floor-clipped, rectified dB
        x = 20 * np.log10(np.abs(x + 1e-10))

        # Initialize slides, assumes that first half second is silent
        for i in range(
            int(0.1 * self.sr), int(0.5 * self.sr), self.block_size
        ):
            xi = x[i : i + self.block_size]
            self.fast_slide(xi)
            self.slow_slide(xi)

        rel = np.zeros_like(x)
        for i in range(0, len(x), self.block_size):
            xi = x[i : i + self.block_size]
            rel[i : i + self.block_size] = self.fast_slide(
                xi
            ) - self.slow_slide(xi)

        self.mins = np.median(rel[: self.sr], axis=0)
        self.maxs = np.max(rel, axis=0)
        self.on_threshold = self.maxs * self.on_threshold + self.mins
        self.off_threshold = self.maxs * self.off_threshold + self.mins
        self.noise_max = np.median(
            maximum_filter1d(rel[::], int(self.sr * 0.01), axis=0), axis=0
        )
        noise_thresh = (self.noise_max - self.mins) / self.maxs
        print(
            "Approx. relative noise thresholds at "
            f"{[np.round(x, 3) for x in noise_thresh]}!"
        )

        # Ensure continuity with the starting point again
        x = x[self.sr - 1 :: -1].copy()
        for i in range(0, self.sr, self.block_size):
            xi = x[i : i + self.block_size]
            self.fast_slide(xi)
            self.slow_slide(xi)

