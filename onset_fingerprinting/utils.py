from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def clipping_audio(x: np.ndarray, labels: pd.DataFrame) -> set:
    """Return examples where input audio is clipping.

    :param x: audio normalized to -1, 1
    :param labels: label dataframe
    """
    bad_idx = np.where((x == 1) | (x == -1))[0]
    bad_examples = pd.IntervalIndex(
        pd.IntervalIndex.from_arrays(labels.start, labels.end)
    ).get_indexer(bad_idx)
    return set(bad_examples)


def knn_metrics(X_test, y_train, y_test, knn: KNeighborsClassifier):
    """Generate KNN metrics for a given test dataset:

    For each example x in X_test and k classes, where k is the number of true
    examples of the class of x (in y_test) in the training dataset (y_train):

        1. Distance of x to the closest k neighbors given knn

        2. The cumulative percentage of the closest neighbors that have the
           same class as x, from 1 to k

    :param X_test: test dataset
    :param y_train: training labels
    :param y_test: test labels
    :param knn: trained kNNClassifier instance
    """

    classes = np.unique(y_test)
    res = {}
    for c in classes:
        idx = y_test == c
        cl = y_test[idx]
        n_c = len(cl)
        X = X_test[idx]
        dist, neigh = knn.kneighbors(X, n_c)
        correct = np.cumsum(y_train[neigh] == c, axis=1) / (np.arange(n_c) + 1)
        res[c] = (dist, correct)
    return res


def plot_res(x, knn, labels, c):
    """

    :param x: Single example with N features
    :param knn:  KNN trained on M examples, each having N features
    :param labels: integer labels for M training examples
    :param c: true class of x
    """

    dist, neigh = knn.kneighbors(x, knn.n_samples_fit_)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.plot(dist[0], label="Distance of nth neighbor")
    ax2 = ax.twinx()
    ax2.plot(
        np.cumsum(labels[neigh[0]] == c) / (np.arange(knn.n_samples_fit_) + 1),
        color="orange",
        label="Correct classification (cumulative)",
    )
    fig.legend()


def plot_knn_metrics(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: None | list[str] = None,
    plot_size: int = 3,
):
    """Plot results of knn_metrics in one plot.  Uses one column per class in
    results, so make sure to have ample of horizontal space!

    :param results: Output of knn_metrics
    :param labels: String labels corresponding to the integer classes/keys of
        results
    """
    if labels is None:
        labels = list(results.keys())

    n = len(labels)
    fig, axs = plt.subplots(
        1, n, sharey=True, figsize=(plot_size * n, plot_size)
    )
    fig.suptitle(
        "Average distance vs correct classification per number of neighbors"
    )
    for c, label, ax in zip(results, labels, axs):
        result = results[c]
        n_neighbors = len(result[0])
        index = np.tile(np.arange(n_neighbors), n_neighbors)
        sns.lineplot(
            pd.Series(result[0].flatten(), index=index),
            label="Distance of nth neighbor",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Number of neighbors")
        ax.set_ylabel("Distance")
        ax2 = ax.twinx()
        sns.lineplot(
            pd.Series(result[1].flatten(), index=index),
            color="orange",
            label="Correct classification (cumulative)",
            ax=ax2,
            legend=False,
        )
        ax2.set_ylim((0, 1))
        ax2.set_ylabel("Percent correctly classified")
        plt.title(f"Class {label}")
        if (label == labels[0]) and len(axs) > 1:
            ax2.set(yticklabels=[], ylabel=None)
    fig.tight_layout()


def compare_model_confusion(
    test_labels: np.ndarray, pred_labels: list, normalize=None, psize=4
):
    n = len(pred_labels)
    fig, axs = plt.subplots(1, n, figsize=(n * psize, psize))

    labels = list(set(test_labels) | set.union(*[set(l) for l in pred_labels]))
    for pred, ax in zip(pred_labels, axs):
        ConfusionMatrixDisplay.from_predictions(
            test_labels, pred, labels=labels, ax=ax, xticks_rotation="vertical"
        )
    fig.tight_layout()


def plot_disagreements(test_labels, predicted_labels_list):
    num_models = len(predicted_labels_list)
    num_samples = len(test_labels)
    labels = list(
        set(test_labels) | set.union(*[set(l) for l in predicted_labels_list])
    )
    num_labels = len(labels)
    ld = {l: i for l, i in zip(labels, range(num_labels))}

    # Prepare an array where rows represent models and columns represent instances
    agreement_array = np.empty((num_models + 1, num_samples))
    agreement_array[0, :] = np.vectorize(ld.get)(test_labels)

    # Fill the array: 1 for correct prediction, 0 for incorrect
    for i, preds in enumerate(predicted_labels_list):
        agreement_array[i + 1, :] = preds == test_labels

    # Find instances that at least one model misclassified
    misclassified = np.any(agreement_array[1:] == 0, axis=0)

    for i, preds in enumerate(predicted_labels_list):
        agreement_array[i + 1, :] = np.vectorize(ld.get)(preds)

    sorter = np.lexsort(agreement_array[::-1, :], axis=0)
    misclassified = misclassified[sorter]
    agreement_array = agreement_array[:, sorter]
    # Create a colormap with a unique color for each label
    cmap = ListedColormap(sns.color_palette(n_colors=num_labels))

    # Create a figure
    fig = plt.figure(figsize=(10, num_models))

    # Use imshow to create a heatmap, only for misclassified instances
    plt.imshow(agreement_array[:, misclassified], aspect="auto", cmap=cmap)

    # Configure the axes and labels
    plt.yticks(
        np.arange(num_models + 1),
        ["True"] + [f"Model {i+1}" for i in range(num_models)],
    )
    plt.xticks([])
    plt.title("Model Disagreements on Misclassified Instances")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap.colors[i])
        for i in range(num_labels)
    ]
    fig.legend(
        handles,
        labels,
        ncols=num_labels,
        fontsize="small",
        columnspacing=0.5,
        loc="upper center",
        bbox_to_anchor=(0.44, 0.1),
    )
    fig.tight_layout()


def plot_misclf(
    true_labels: np.ndarray,
    pred_labels: list[np.ndarray],
    psize: float = 1.2,
    model_names: list[str] | None = None,
    normalize: bool = False,
):
    n = len(pred_labels)
    if model_names is None:
        model_names = list(map(str, range(len(pred_labels))))
    else:
        assert len(model_names) == n

    labels = list(set(true_labels) | set.union(*[set(l) for l in pred_labels]))
    n_classes = len(labels)
    n_models = len(pred_labels)

    cm = np.zeros((n_models, n_classes, n_classes))
    for i, preds in enumerate(pred_labels):
        cm[i] = confusion_matrix(true_labels, preds, labels=labels)

    df = pd.concat((pd.DataFrame(x, columns=labels, index=labels) for x in cm))
    model = [[i] * len(cm[i]) for i in range(n)]
    df["model"] = [x for y in model for x in y]

    df = df.melt(
        df.columns[-1],
        var_name="pred",
        value_name="misclassifications",
        ignore_index=False,
    ).reset_index(names="true")

    df = df[(df["misclassifications"] != 0) & (df["true"] != df["pred"])]
    if normalize:
        df["misclassifications"] /= df.groupby("model")[
            "misclassifications"
        ].transform(sum)
    trues = df["true"].unique()
    preds = df["pred"].unique()
    df = df.set_index(["true", "pred"])

    fig, axs = plt.subplots(
        len(preds),
        len(trues),
        figsize=(len(trues) * psize, len(preds) * psize),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cp = np.array(sns.color_palette(n_colors=n))
    for i, (axs_row, pred) in enumerate(zip(axs, preds)):
        for j, (ax, true) in enumerate(zip(axs_row, trues)):
            try:
                row = df.loc[true, pred]
                ax.bar(
                    row["model"],
                    row["misclassifications"],
                    1,
                    color=cp[row["model"]],
                )
            except:
                # for spine in ax.spines:
                #     ax.spines[spine].set_visible(False)
                pass
            if i == len(preds) - 1:
                ax.set_xlabel(true)
            if j == 0:
                ax.set_ylabel(pred)
            ax.set_xticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=cp[i]) for i in range(n)]
    fig.legend(handles, [m for m in model_names], title="Model")


# https://stackoverflow.com/questions/39032325/python-high-pass-filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def wave_speed(T0: float, rho0: float):
    """
    Calculate the wave speed in a vibrating membrane.

    Reference: Fletcher, N.  H., & Rossing, T.  D.  (1998).  The Physics of
    Musical Instruments (2nd ed.).  Springer.

    :param T0: Tension in the membrane in N/m.
    :param rho0: Areal density of the membrane in kg/m^2.

    :returns: Speed of wave propagation in m/s.
    """
    return np.sqrt(T0 / rho0)


def drum_frequency(diameter_m: float, T0: float, rho0: float, m: int, n: int):
    """
    Calculate the frequency of a specific mode of a circular drum.

    Reference: Fletcher, N.  H., & Rossing, T.  D.  (1998).  The Physics of
    Musical Instruments (2nd ed.).  Springer.

    According to TPoMI, where they measured a 32cm Tom Ambassador drumhead to
    weigh 50g, Ambassador drumheads would sit at an areal density of around
    0.05kg/m^2.  It also states that 351N/m would be a low value of tension for
    a 32cm tom.

    :param diameter_m: Diameter of the drum in meters.
    :param T0: Tension in the membrane in N/m.
    :param rho0: Areal density of the membrane in kg/m^2.
    :param m: Radial mode number.
    :param n: Azimuthal mode number.

    :returns: Frequency of the drum for the given mode in Hz.
    """
    D = diameter_m
    v = wave_speed(T0, rho0)
    # Wavenumber in 1/m
    k = np.sqrt(m**2 + n**2) * np.pi / D

    return v * k / (2 * np.pi)
