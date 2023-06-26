import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

mel_args = {"fmin": 200, "fmax": 12000, "n_mels": 40, "sr": 44100}
mfcc_args = {"n_mfcc": 14}


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

    def __init__(self, path: str | Path, channel: str):
        """Initialize POSD.

        Future args?
        - recompute velocity
        - improve onsets

        :param path: path to folder containing the dataset
        :param channel: name of channel to load
        """
        # go into path, recursively load all matching files
        path = Path(path)
        hit_meta_files = list(path.rglob("*_hits.json"))
        session_meta_files = [x.with_stem(x.stem[:-5]) for x in hit_meta_files]
        sessions = [read_json(x) for x in session_meta_files]
        assert all(channel in x["channels"] for x in sessions)
        hits = [parse_hits(read_json(x)) for x in hit_meta_files]
        files = [
            x.with_name(x.stem + f"_{channel}.wav") for x in session_meta_files
        ]

    def load_files(files, session_meta, hits):
        for file, meta in zip(files, session_meta):
            audio, sr = sf.read(file)

    def get_onset_frames(audio, onsets):
        pass


def window_contribution_weight(window: np.ndarray, hop_length: int):
    """Create an array of stft frame weights which corresponds to the amount of
    the signal of interest which contributed to the frame due to windowing.

    :param window: window multiplied with audio frames in the STFT
    :param hop_length: hop_length of the stft
    """
    w = []
    for i in range(1, 1 + len(window) // hop_length):
        w.append(np.trapz(window[: i * hop_length]))
    return np.concatenate((w, w[i - 2 :: -1])) / max(w)


def stft_frame(x: np.ndarray, n_fft: int, window: np.ndarray):
    """Compute single STFT frame using the fft.

    :param x: audio to transform, border-padded but not for oversampling
    :param n_fft: size of the fft
    :param window: fft windowing function array of size n_fft
    """
    if n_fft > len(x):
        x = librosa.util.pad_center(x, size=n_fft)
    return np.fft.rfft(window * x)


def mfcc(
    audio: np.ndarray,
    onset: int,
    frame_length: int = 256,
    hop_length: int = 64,
    n_fft: int = 512,
    flucoma: bool = False,
    method="zerozero",
    mel_args=mel_args,
    mfcc_args=mfcc_args,
):
    """Compute mel-frequency cepstral coefficients (MFCCs) around a given onset
    inside an audio array.

    :param onset: index into audio at which the onset begins
    :param audio: mono audio array containing onsets
    :param frame_length: frame length for the STFT
    :param hop_length: hop length for the STFT
    :param n_fft: size of the FFT, can be larger than frame_length in case
        oversampling is desired
    :param flucoma: whether to use additional padding such that the first and
        last stft frames use hop_length audio samples respectively
    :param method: zerozero: pad front and back of audio with zeros, prezero:
        pad front with preceding audio, back with zeros, pre: pad front with
        preceding audio, don't pad back
    """
    y = audio[onset : onset + frame_length]
    pad_length = frame_length - hop_length if flucoma else frame_length // 2
    pad = np.zeros(pad_length, dtype=np.float32)
    pre = audio[onset - pad_length : onset]
    window = librosa.filters.get_window("hann", frame_length, fftbins=True)
    if n_fft > frame_length:
        window = librosa.util.pad_center(window, size=n_fft)

    if method == "zerozero":
        # 1: start at onset, zero both ends
        y = np.concatenate((pad, y, pad))

    elif method == "prezero":
        # 2: start at onset, zero end, take audio before

        y = np.concatenate((pre, y, pad))

    elif method == "pre":
        # 3: take frames from ongoing STFT, hard end on last frame
        y = np.concatenate((pre, y))

    n_frames = 1 + (len(y) - frame_length) // hop_length
    S = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        S[:, i] = stft_frame(
            y[hop_length * i : hop_length * i + frame_length], n_fft, window
        )
    mels = librosa.feature.melspectrogram(S=np.abs(S) ** 2, **mel_args)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mels), **mfcc_args)
    return mfcc, S
