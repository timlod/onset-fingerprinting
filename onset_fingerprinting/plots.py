import matplotlib.pyplot as plt
from onset_fingerprinting import multilateration
import numpy as np


def polar_circle(polar_coords: list[tuple[float, float]]) -> None:
    """
    Plot a unit circle and scatter a list of polar coordinates on it.

    :param polar_coords: List of tuples, each containing radius and angle in degrees.
    """
    # Plot unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.sin(theta)
    y_circle = np.cos(theta)
    plt.plot(x_circle, y_circle, label="Unit Circle")

    x_values = np.zeros(len(polar_coords))
    y_values = np.zeros(len(polar_coords))
    for i, (r, angle) in enumerate(polar_coords):
        x_values[i] = r * np.sin(np.radians(angle))
        y_values[i] = r * np.cos(np.radians(angle))

    # Scatter plot with colormap
    plt.scatter(
        x_values, y_values, c=range(len(polar_coords)), cmap="coolwarm"
    )

    # Set aspect ratio and labels
    plt.axis("equal")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("Circle and Scatter Plot")


def is_legal_3d_plot(m, group, tolerance=2):
    # We take a tolerance of 2cm around the target for this
    if tolerance is None:
        tolerance = m.samples_per_cm * 1
    else:
        tolerance *= m.samples_per_cm
    sensors, onsets = group[0], group[1]
    lag1 = onsets[1] - onsets[0]
    lag2 = onsets[2] - onsets[0]
    lag3 = onsets[2] - onsets[1]
    print(tolerance, m.samples_per_cm, lag1, lag2)
    lm1 = m.lag_maps[sensors[0]][sensors[1]]
    lm2 = m.lag_maps[sensors[0]][sensors[2]]
    lm3 = m.lag_maps[sensors[1]][sensors[2]]
    plt.imshow(lm1)
    plt.colorbar()
    plt.figure()
    plt.imshow(lm2)
    plt.colorbar()
    plt.figure()
    plt.imshow(lm3)
    plt.colorbar()
    plt.figure()
    legal = (lm1 < lag1 + tolerance) & (lm1 > lag1 - tolerance)
    plt.imshow(legal)
    legal &= (lm2 < lag2 + tolerance) & (lm2 > lag2 - tolerance)
    plt.figure()
    plt.imshow(legal)
    legal &= (lm3 < lag3 + tolerance) & (lm3 > lag3 - tolerance)
    plt.figure()
    plt.imshow(legal)
    return np.unravel_index(np.argmax(legal > 0), legal.shape, "F")


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
