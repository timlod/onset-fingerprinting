import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
