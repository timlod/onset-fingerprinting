import matplotlib.pyplot as plt
from onset_fingerprinting import echolocation
import numpy as np


def plot_around(x, peaks, i, n=256, hop=32):
    peak = peaks[i]
    l = peak - n // 2
    r = peak + n // 2
    plt.plot(x[l:r])
    plt.vlines(peak - l, 0, x[l:r].max(), "r")
    plt.vlines(peak - l + n // 2, 0, x[l:r].max(), "g")
    plt.vlines(peak - l + n // 2 - hop, 0, x[l:r].max(), "y")


def plot_lags_2D(mic_a, mic_b, d=14 * 2.54, sr=96000):
    """Plot

    :param mic_a:
    :param mic_b:
    :param d:
    :param sr:
    :returns:

    """
    n = int(np.round(d, 1) * 10)
    r = n // 2
    mic_a = echolocation.polar_to_cartesian(mic_a[0] * r, mic_a[1])
    mic_b = echolocation.polar_to_cartesian(mic_b[0] * r, mic_b[1])
    lags = echolocation.sim_lag_centered(mic_a, mic_b, d, sr)

    plt.imshow(lags, cmap="RdYlGn", extent=[-r, r, -r, r])
    plt.colorbar()
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


def plot_lags_3d(mic_a, mic_b, d=14 * 2.54, sr=96000):
    n = int(np.round(d, 1) * 10)
    r = n // 2
    mic_a_cart = echolocation.spherical_to_cartesian(
        mic_a[0] * r, mic_a[1], mic_a[2]
    )
    mic_b_cart = echolocation.spherical_to_cartesian(
        mic_b[0] * r, mic_b[1], mic_b[2]
    )
    lags, sa, sb = echolocation.sim_3d(mic_a_cart, mic_b_cart, d, sr)
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
