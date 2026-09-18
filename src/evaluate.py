"""One evaluator for every row of the comparison tables.

The report does not score every model the same way, and this module reproduces
what each row actually used rather than imposing one convention. The arguments
that select a published row:

    row in the report            genres                 average  auroc_on  in_test
    AlexNet, ResNet-18, -50      THREE_GENRES_TRANSFER  macro    label     True
    YOLO multi-class             THREE_GENRES_TRANSFER  micro    score     False
    CLIP, EfficientNetV2, ViT    THREE_GENRES_ADVANCED  macro    score     False
    nineteen-genre tables        None                   as above  as above  as above

auroc_on="label" feeds the hard arg max to roc_auc_score, which is what
evaluate18, evaluate50 and evaluateAlex did. auroc_on="score" feeds the
continuous scores. The two are not the same quantity and the conclusion tables
mix them.

in_test=False scores every row of the prefix supplied rather than the test half,
which is what the YOLO evaluation did.
"""

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             roc_auc_score, roc_curve)

import config
from data_io import genre_names, split_indices


def load_predictions(path: str) -> np.ndarray:
    return np.load(path)["arr_0"]


@lru_cache(maxsize=1)
def _truth():
    """The genre table, its one-hot block and the genre names. Read once."""
    table = pd.read_csv(config.ENCODED_GENRES_CSV)
    return table, table.iloc[:, 1:-1].to_numpy(), genre_names(table.columns[1:-1])


def selection_mask(n_rows: int, genres: list = None,
                   in_test: bool = True) -> np.ndarray:
    """Single-genre rows, optionally restricted to a triple and to the test half."""
    table, truth, names = _truth()
    mask = (table[config.GENRE_COUNT].to_numpy()[:n_rows] == 1)
    if in_test:
        _, test_indices = split_indices(len(table))
        mask &= np.isin(np.arange(n_rows), test_indices)
    if genres is not None:
        columns = np.isin(names, genres)
        mask &= truth[:n_rows][:, columns].sum(axis=1) >= 1
    return mask


def metrics(predictions: np.ndarray, genres: list = None, average: str = "macro",
            auroc_on: str = "label", in_test: bool = True) -> dict:
    """Accuracy, F1, precision and AUROC for one model. See the module docstring."""
    table, truth, names = _truth()
    columns = np.isin(names, genres) if genres is not None \
        else np.ones(len(names), bool)
    mask = selection_mask(len(predictions), genres, in_test)

    truth = truth[:len(predictions)][mask][:, columns]
    scores = predictions[mask][:, columns]
    predicted = scores == scores.max(axis=1)[:, None]
    shares = truth.sum(axis=0) / truth.shape[0]

    return {
        "n": int(mask.sum()),
        "baseline": float(shares.max()),
        "accuracy": accuracy_score(truth, predicted),
        "f1": f1_score(truth, predicted, average=average, zero_division=0),
        "precision": precision_score(truth, predicted, average=average,
                                     zero_division=0),
        "roc_auc": roc_auc_score(truth, predicted if auroc_on == "label" else scores,
                                 average=average),
    }


def auroc_by_genre(predictions: np.ndarray, indices: np.ndarray = None,
                   plot: bool = False) -> pd.DataFrame:
    """Per-genre AUROC on the raw scores, as the section 4.2 table reports it."""
    _, truth, names = _truth()
    if indices is not None:
        truth, predictions = truth[indices], predictions[indices]
    scores = {name: roc_auc_score(truth[:, i], predictions[:, i])
              for i, name in enumerate(names)}
    if plot:
        for i, name in enumerate(names):
            fpr, tpr, _ = roc_curve(truth[:, i], predictions[:, i])
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Baseline")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Multilabel ROC curve")
        plt.legend()
    return pd.DataFrame(scores.items(), columns=["Genre", "ROC AUC"])
