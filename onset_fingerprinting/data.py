import json
from pathlib import Path
from typing import Callable

import audiomentations
import fcwt
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample
from torch.utils.data import Dataset

## TODO: listen to examples which are constantly misclassified to check if I
## could tell them apart myself

AUGMENTATIONS = [
    audiomentations.AddGaussianNoise(p=1),
    audiomentations.AirAbsorption(p=1),
    audiomentations.SevenBandParametricEQ(
        min_gain_db=-10, max_gain_db=10, p=1
    ),
    # audiomentations.Gain(p=1),
    audiomentations.TanhDistortion(
        min_distortion=0.005, max_distortion=0.1, p=1
    ),
]


def read_json(file: Path) -> dict:
    """Read JSON file and return its dictionary.

    :param file: path to json file
    """
    with open(file, "r") as f:
        return json.load(f)


def parse_hits(d: dict) -> pd.DataFrame:
    """Parse hits dictionary into dataframe.

    Mainly unwraps the conditions list such that we can call pd.DataFrame(d).

    :param d: dictionary output of json.load(<hits_file>)
    """
    d = d.copy()
    if "conditions" in d:
        for cond in d["conditions"]:
            d[cond] = d["conditions"][cond]
        del d["conditions"]
    return pd.DataFrame(d)


class POSD(Dataset):
    """PyTorch onset audio classification dataset."""

    def __init__(
        self,
        path: str | Path,
        frame_length: int,
        channel: str,
        # e.g. lambda x, posd: cspec_to_mfcc(stft(x, posd.pre_samples), sr=sr)
        transform: Callable | None = None,
        pre_samples: int = 0,
        augmentations: list = AUGMENTATIONS,
        n_rounds_aug: int = 5,
    ):
        """Initialize POSD.

        Future args?

            - recompute velocity

            - improve onsets

        The transformation will need to be composable on individual examples I
        guess.

        Idea: extract slightly larger window than frame around each onset to
        use in transformations.  This way we don't need to keep the entire
        audio file, but can still use the neighborhood of the onset to do data
        augmentation.  Alternatively, pre-compute a sane number of
        augmentations beforehand using aug_transform - these will be done while
        loading the original file to keep memory use to a minimum.  The
        reasoning here is that computing full augmentation and transformation
        for single 256 sample bits is going to be the bottleneck in training.
        With this approach, we don't add randomness at every epoch, but just
        increase the dataset size usable.

        :param path: path to folder containing the dataset
        :param channel: name of channel to load
        :param transform: function which takes as input an array of audio and a
            sequence of onset start indexes, and returns an array of
            transformed data
        :param aug_transform: function which takes as input an array of audio
            and a sequence of onset start indexes, and returns modified audio
        :param pre_samples: take this many samples from before the onset
        :param n_rounds_aug: how many times to duplicate training data under a
            different set of augmentations.  This is done for each frame
            extractor
        """
        # go into path, recursively load all matching files
        path = Path(path)
        hit_meta_files = list(path.rglob("*_hits.json"))
        session_meta_files = [x.with_stem(x.stem[:-5]) for x in hit_meta_files]
        sessions = [read_json(x) for x in session_meta_files]
        assert all(channel in x["channels"] for x in sessions)
        self.sessions = sessions
        self.hits = [parse_hits(read_json(x)) for x in hit_meta_files]
        self.files = [
            x.with_name(x.stem + f"_{channel}.wav") for x in session_meta_files
        ]

        self.frame_length = frame_length
        self.pre_samples = pre_samples
        self.frame_extractor = FrameExtractor(frame_length, pre_samples)
        self.extra_extractors = [
            self.frame_extractor,
            SampleShiftFrameExtractor(frame_length, pre_samples, 6),
            StretchFrameExtractor(frame_length, pre_samples, 0.06),
        ]
        self.aug = audiomentations.SomeOf((0, 3), augmentations, p=1)
        self.n_rounds_aug = n_rounds_aug

        self.transform = transform
        self.load_files()
        if self.transform is not None:
            self.audio = self.transform(self.audio, self)

    def load_files(self):
        files, sessions, hits_per_session = (
            self.files,
            self.sessions,
            self.hits,
        )
        self.audio = np.empty(
            (
                (1 + len(self.extra_extractors) * self.n_rounds_aug)
                * sum(len(h) for h in hits_per_session),
                self.frame_length + self.pre_samples,
            ),
            dtype=np.float32,
        )
        self.labels = []
        i = 0
        for file, session, hits in zip(files, sessions, hits_per_session):
            self.labels.extend(hits.zone)
            audio, sr = sf.read(file)
            # This is the raw data
            self.audio[i : i + len(hits)] = self.frame_extractor(
                audio, hits.onset_start
            )
            # Everything here will be augmented
            if self.n_rounds_aug > 0:
                for extractor in self.extra_extractors:
                    aug_audio = extractor(audio, hits.onset_start)
                    for _ in range(self.n_rounds_aug):
                        self.labels.extend(hits.zone)
                        i += len(hits)
                        self.audio[i : i + len(hits)] = self.aug(
                            aug_audio.T, sr
                        ).T

            i += len(hits)


class FrameExtractor:
    """
    Given a full audio waveform and onsets of interest, select for each onset
    the frame of interest.
    """

    def __init__(self, frame_length: int, pre_samples: int):
        self.frame_length = frame_length
        self.pre_samples = pre_samples

    def __call__(self, audio: np.ndarray, onsets: np.ndarray):
        view = np.lib.stride_tricks.sliding_window_view(
            audio, window_shape=(self.frame_length + self.pre_samples)
        )
        return view[onsets - self.pre_samples]


class SampleShiftFrameExtractor(FrameExtractor):
    def __init__(self, frame_length: int, pre_samples: int, max_shift: int):
        super().__init__(frame_length, pre_samples)
        self.max_shift = max_shift

    def __call__(self, audio: np.ndarray, onsets: np.ndarray):
        shifts = np.random.randint(1, self.max_shift, len(onsets))
        np.negative(
            shifts,
            out=shifts,
            where=np.random.randint(2, size=len(shifts), dtype=bool),
        )
        view = np.lib.stride_tricks.sliding_window_view(
            audio, window_shape=(self.frame_length + self.pre_samples)
        )
        return view[onsets + shifts - self.pre_samples]


class StretchFrameExtractor(FrameExtractor):
    def __init__(
        self, frame_length: int, pre_samples: int, max_stretch: float = 0.03
    ):
        super().__init__(frame_length, pre_samples)
        self.max_shift = int((frame_length + pre_samples) * max_stretch)
        self.full_length = frame_length + pre_samples

    def __call__(self, audio, onsets):
        shifts = np.random.randint(1, self.max_shift, len(onsets))
        np.negative(
            shifts,
            out=shifts,
            where=np.random.randint(2, size=len(shifts), dtype=bool),
        )
        out = np.empty((len(onsets), self.full_length), dtype=np.float32)
        for i, (onset, shift) in enumerate(
            zip(onsets - self.pre_samples, shifts)
        ):
            out[i] = resample(
                audio[onset : onset + self.full_length + shift],
                self.full_length,
            )
        return out


def window_contribution_weights(window, hop_length, hop_edge_padding: False):
    """Create an array of stft frame weights which corresponds to the amount of
    the signal of interest which contributed to the frame due to windowing.

    :param window: window multiplied with audio frames in the STFT
    :param hop_length: hop_length of the stft
    :param hop_edge_padding: whether hop_edge padding was used in stft
    """
    w = []
    start_idx = len(window) // 2 if not hop_edge_padding else hop_length

    for i in range(start_idx, len(window) + hop_length, hop_length):
        w.append(np.trapz(window[:i]))
    w += w[-2::-1]
    return np.array(w) / max(w)


def stft_frame(x: np.ndarray, n_fft: int, window: np.ndarray):
    """Compute single STFT frame using the fft.

    :param x: audio to transform, border-padded but not for oversampling
    :param n_fft: size of the fft
    :param window: fft windowing function array of size n_fft
    """
    if n_fft > x.shape[-1]:
        x = librosa.util.pad_center(x, size=n_fft)
    return np.fft.rfft(window * x)


def stft(
    audio: np.ndarray,
    onset: int,
    frame_length: int = 256,
    hop_length: int = 64,
    n_fft: int = 512,
    hop_edge_padding: bool = False,
    method="zerozero",
):
    """Compute mel-frequency cepstral coefficients (MFCCs) around a given onset
    inside an audio array.

    :param onset: index into audio at which the onset begins
    :param audio: mono audio array containing onsets
    :param frame_length: frame length for the STFT
    :param hop_length: hop length for the STFT
    :param n_fft: size of the FFT, can be larger than frame_length in case
        oversampling is desired
    :param hop_edge_padding: whether to use additional padding such that the
        first and last stft frames use hop_length audio samples respectively.
        This is the approach used in FluCoMa.  If this is not used, the STFT
        will reflect how a usual (say librosa) centered STFT would work, with
        the first frame containing at least half of the signal.
    :param method: zerozero: pad front and back of audio with zeros, prezero:
        pad front with preceding audio, back with zeros, pre: pad front with
        preceding audio, don't pad back
    """
    y = audio[..., onset : onset + frame_length]
    pad_length = (
        frame_length - hop_length if hop_edge_padding else frame_length // 2
    )
    dim0 = 1 if y.ndim == 1 else y.shape[0]
    pad = np.zeros((dim0, pad_length), dtype=np.float32).squeeze()
    pre = audio[..., onset - pad_length : onset]
    window = librosa.filters.get_window("hann", frame_length, fftbins=True)
    if n_fft > frame_length:
        window = librosa.util.pad_center(window, size=n_fft)

    if method == "zerozero":
        # 1: start at onset, zero both ends
        y = np.concatenate((pad, y, pad), axis=-1)

    elif method == "prezero":
        # 2: start at onset, zero end, take audio before

        y = np.concatenate((pre, y, pad), axis=-1)

    elif method == "pre":
        # 3: take frames from ongoing STFT, hard end on last frame
        y = np.concatenate((pre, y), axis=-1)

    n_frames = 1 + (y.shape[-1] - frame_length) // hop_length
    S = np.empty(
        (dim0, n_fft // 2 + 1, n_frames), dtype=np.complex64
    ).squeeze()
    for i in range(n_frames):
        S[..., i] = stft_frame(
            y[..., hop_length * i : hop_length * i + frame_length],
            n_fft,
            window,
        )
    return S


def cspec_to_mfcc(
    S: np.ndarray,
    sr: int,
    fmin: int = 0,
    fmax: None | int = None,
    n_mels: int = 40,
    n_mfcc: int = 14,
):
    """Compute MFCCs from a complex spectrogram.

    :param S: spectrogram output of stft, complex
    :param sr: sample rate
    :param fmin: lowest frequency
    :param fmax: highest frequency
    :param n_mels: number of mel bands to use
    :param n_mfccs: number of mfccs to compute
    """
    mels = librosa.feature.melspectrogram(
        S=np.abs(S) ** 2, sr=sr, fmin=fmin, fmax=fmax, n_mels=n_mels
    )
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mels), sr=sr, n_mfcc=n_mfcc
    )
    return mfcc
