import librosa
import numpy as np

mel_args = {"fmin": 200, "fmax": 12000, "n_mels": 40, "sr": 44100}
mfcc_args = {"n_mfcc": 14}


def stft_frame(x: np.ndarray, n_fft: int, window: np.ndarray):
    """Compute single STFT frame using the fft.

    :param x: audio to transform, border-padded but not for oversampling
    :param n_fft: size of the fft
    :param window: fft windowing function array of size n_fft
    """
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
            y[hop_length * i : hop_length * i + frame_length],
            n_fft,
            window,
        )
    mels = librosa.feature.melspectrogram(S=np.abs(S) ** 2, **mel_args)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mels), **mfcc_args)
    return mfcc
