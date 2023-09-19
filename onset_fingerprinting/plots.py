import matplotlib.pyplot as plt
from onset_fingerprinting import multilateration
import numpy as np


def plot_around(x, peaks, i, n=256, hop=32, only_peak=True):
    peak = peaks[i]
    left = peak - n // 2
    right = peak + n // 2
    plt.plot(x[left:right])
    plt.vlines(
        peak - left,
        x[left:right].min(),
        x[left:right].max(),
        "r",
        label=f"Peak {i}",
    )
    if not only_peak:
        plt.vlines(
            peak - left + hop,
            0,
            x[left:right].min(),
            x[left:right].max(),
            "orange",
            label=f"Peak {i} + hop ({hop})",
        )
        plt.vlines(
            peak - left + n // 2, x[left:right].min(), x[left:right].max(), "g"
        )
        plt.vlines(
            peak - left + n // 2 - hop,
            x[left:right].min(),
            x[left:right].max(),
            "y",
        )


def plot_lags_2D(
    mic_a: tuple[int, int],
    mic_b: tuple[int, int],
    d: int = multilateration.DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = multilateration.MEDIUM,
):
    """Plot lag map for 2D mic locations.

    :param mic_a: location of microphone A, in cartesian coordinates
    :param mic_b: location of microphone A, in cartesian coordinates
    :param d: diameter of the drum, in centimeters
    :param sr: sampling rate
    :param scale: scale to increase/decrease precision originally in
        centimeters.  For example, for millimeters, scale should be 10
    :param medium: the medium the sound travels through.  One of 'air' or
        'drumhead', the latter for optical/magnetic measurements
    """
    n = int(np.round(d, 1) * 10)
    r = n // 2
    mic_a = multilateration.polar_to_cartesian(mic_a[0] * r, mic_a[1])
    mic_b = multilateration.polar_to_cartesian(mic_b[0] * r, mic_b[1])
    lags = multilateration.lag_map_2d(mic_a, mic_b, d, sr, scale, medium)

    plt.imshow(lags, cmap="RdYlGn", extent=[-r, r, -r, r])
    plt.colorbar(label="Samples difference")
    plt.scatter(
        mic_a[0],
        -mic_a[1],
        marker="o",
        label="Mic A",
        c="white",
        edgecolors="black",
    )
    plt.scatter(
        mic_b[0],
        -mic_b[1],
        marker="o",
        label="Mic B",
        c="black",
        edgecolors="white",
    )
    circle = plt.Circle((0, 0), r, edgecolor="black", facecolor="none")
    plt.gca().add_artist(circle)
    plt.legend()


def plot_lags_3d(
    mic_a,
    mic_b,
    reflectivity: float = 0.5,
    d: int = multilateration.DIAMETER,
    sr: int = 96000,
    scale: float = 1,
    medium: str = "air",
):
    """Plot lags and sound intensities dropoffs for 3D mic locations.

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
    n = int(np.round(d, 1) * scale)
    r = n // 2
    mic_a_cart = multilateration.spherical_to_cartesian(
        mic_a[0] * r, mic_a[1], mic_a[2]
    )
    mic_b_cart = multilateration.spherical_to_cartesian(
        mic_b[0] * r, mic_b[1], mic_b[2]
    )
    lags, sa, sb = multilateration.lag_intensity_map(
        mic_a_cart, mic_b_cart, reflectivity, d, sr, scale, medium
    )
    plot_heatmap(lags, r, mic_a_cart, mic_b_cart, "Samples difference")
    plot_heatmap(sa, r, mic_a_cart, mic_b_cart, "dB")
    plot_heatmap(sb, r, mic_a_cart, mic_b_cart, "dB")


def plot_heatmap(x, r, mic_a_cart, mic_b_cart, cb_label=""):
    plt.figure()
    plt.imshow(x, cmap="RdYlGn", extent=[-r, r, -r, r])
    plt.colorbar(label=cb_label)
    plt.scatter(
        mic_a_cart[0],
        -mic_a_cart[1],
        marker="o",
        label="Mic A",
        c="white",
        edgecolors="black",
    )
    plt.scatter(
        mic_b_cart[0],
        -mic_b_cart[1],
        marker="o",
        label="Mic B",
        c="black",
        edgecolors="white",
    )

    # Annotate z-component of the microphones
    plt.annotate(
        f"z={mic_a_cart[2]:.1f}",
        (mic_a_cart[0], -mic_a_cart[1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
    )
    plt.annotate(
        f"z={mic_b_cart[2]:.1f}",
        (mic_b_cart[0], -mic_b_cart[1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
    )

    circle = plt.Circle((0, 0), r, edgecolor="black", facecolor="none")
    plt.gca().add_artist(circle)
    plt.legend()
