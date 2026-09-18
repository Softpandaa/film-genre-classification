"""Reading posters and the genre tables from disk."""

import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def get_poster(movie_id) -> np.ndarray:
    """Return the poster of one movie as an HxWx3 uint8 array."""
    return mpimg.imread(os.path.join(config.POSTER_DIR, f"{int(movie_id)}.jpg"))


def load_encoded_genres() -> pd.DataFrame:
    """The one-hot genre table, indexed by movie id, with a genre_count column."""
    return pd.read_csv(config.ENCODED_GENRES_CSV, index_col=0)


def genre_names(columns) -> list:
    """Strip the get_dummies prefix from one-hot column names."""
    return [c[len(config.GENRE_PREFIX):] for c in columns]


def split_indices(n: int):
    """The train and test indices shared by every model and by src/evaluate.py."""
    return train_test_split(np.arange(n), test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)


def single_label_rows(genres: list = None, strict: bool = False,
                      half: str = None) -> pd.DataFrame:
    """Single-label movies, as sections 3 and 4.1 defined them.

    strict=False counts genres only within `genres`, so a movie that is Action
    and Comedy still counts as an Action example. That is what those sections
    ran. strict=True counts over all nineteen, which is how the evaluation
    tables in sections 4.2 and 4.3 select their rows.

    half=None uses the whole corpus, which is what the balanced samplers did.
    """
    table = load_encoded_genres()
    if half is not None:
        train_indices, test_indices = split_indices(len(table))
        table = table.iloc[train_indices if half == "train" else test_indices]
    if strict:
        table = table[table[config.GENRE_COUNT] == 1]
    if genres is not None:
        columns = [config.GENRE_PREFIX + g for g in genres]
        table = table[table[columns].sum(axis=1) == 1]
    return table


def balanced_sample(rows: pd.DataFrame, genres: list, n_per_genre: int,
                    seed: int, check_shape: bool = True) -> pd.DataFrame:
    """Equal numbers per genre, drawn as sections 3 and 4.1 drew them.

    The draw is WITH replacement, by indexing the pool with np.random.randint
    and walking that list until enough posters of the modal shape have been
    accepted. A movie can therefore appear twice, and the duplicates survive
    into the train and test split the caller makes afterwards.
    """
    np.random.seed(seed)
    picked = []
    for genre in genres:
        pool = rows[rows[config.GENRE_PREFIX + genre]]
        draw = np.random.randint(0, len(pool), size=n_per_genre + 10)
        taken, j = [], 0
        while len(taken) < n_per_genre:
            movie_id = pool.index[draw[j]]
            if not check_shape or \
                    sum(get_poster(movie_id).shape) == sum(config.POSTER_SHAPE):
                taken.append(movie_id)
            j += 1
        picked.append(pool.loc[taken])
    return pd.concat(picked)
