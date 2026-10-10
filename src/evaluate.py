"""Metrics of the comparison tables, each row under the convention it used.

    row                             genres                 average  auroc_on  in_test
    AlexNet, ResNet-18, ResNet-50   THREE_GENRES_TRANSFER  macro    label     True
    YOLOv8, both models             THREE_GENRES_TRANSFER  micro    score     False
    CLIP, EfficientNetV2, Fast-ViT  THREE_GENRES_ADVANCED  macro    score     False
    19-genre tables                 None                   as above

auroc_on="label" scores the hard arg max, "score" the raw scores. in_test=False
takes every single-genre row, not only those of the test half.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

import config
from data_io import genre_names, split_indices


def load_predictions(path: str) -> np.ndarray:
    return np.load(path)["arr_0"]


@lru_cache(maxsize=1)
def _truth():
    """The genre table, its one-hot block and the genre names."""
    table = pd.read_csv(config.ENCODED_GENRES_CSV)
    return table, table.iloc[:, 1:-1].to_numpy(), genre_names(table.columns[1:-1])


def selection_mask(n_rows: int, genres: list | None = None, in_test: bool = True) -> np.ndarray:
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


def metrics(predictions: np.ndarray, genres: list | None = None, average: str = "macro",
            auroc_on: str = "label", in_test: bool = True) -> dict:
    """Accuracy, F1, precision and AUROC of one row; see the module docstring."""
    _, truth, names = _truth()
    columns = np.isin(names, genres) if genres is not None else np.ones(len(names), bool)
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
        "precision": precision_score(truth, predicted, average=average, zero_division=0),
        "roc_auc": roc_auc_score(truth, predicted if auroc_on == "label" else scores,
                                 average=average),
    }


def auroc_by_genre(predictions: np.ndarray, indices: np.ndarray | None = None) -> pd.DataFrame:
    """Per-genre AUROC on the raw scores, as Table 20 reports it."""
    _, truth, names = _truth()
    if indices is not None:
        truth, predictions = truth[indices], predictions[indices]
    scores = {name: roc_auc_score(truth[:, i], predictions[:, i])
              for i, name in enumerate(names)}
    return pd.DataFrame(scores.items(), columns=["Genre", "ROC AUC"])
