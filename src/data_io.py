"""Reading posters and the genre tables from disk."""

import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def get_poster(movie_id) -> np.ndarray:
    """The poster of one movie as a uint8 array."""
    return mpimg.imread(os.path.join(config.POSTER_DIR, f"{int(movie_id)}.jpg"))


def load_encoded_genres() -> pd.DataFrame:
    """The one-hot genre table, indexed by movie id, with a genre_count column."""
    return pd.read_csv(config.ENCODED_GENRES_CSV, index_col=0)


def genre_names(columns) -> list:
    """Strip the get_dummies prefix from one-hot column names."""
    return [c[len(config.GENRE_PREFIX):] for c in columns]


def split_indices(n: int):
    """The shared train and test indices of n rows, by position."""
    return train_test_split(np.arange(n), test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)


def single_label_rows(genres: list | None = None, strict: bool = False) -> pd.DataFrame:
    """Movies with exactly one of genres, other genres ignored, as in Sections 3 and 4.1."""
    table = load_encoded_genres()
    if strict:  # one genre over all 19
        table = table[table[config.GENRE_COUNT] == 1]
    if genres is not None:
        columns = [config.GENRE_PREFIX + g for g in genres]
        table = table[table[columns].sum(axis=1) == 1]
    return table


def balanced_sample(rows: pd.DataFrame, genres: list, n_per_genre: int,
                    seed: int, check_shape: bool = True) -> pd.DataFrame:
    """n_per_genre movies per genre, drawn with replacement as Sections 3 and 4.1 drew them."""
    np.random.seed(seed)
    picked = []
    for genre in genres:
        pool = rows[rows[config.GENRE_PREFIX + genre]]
        # 10 spare draws for posters of another shape. A movie drawn twice can fall
        # on both sides of the split the caller makes afterwards.
        draw = np.random.randint(0, len(pool), size=n_per_genre + 10)
        taken, j = [], 0
        while len(taken) < n_per_genre:
            movie_id = pool.index[draw[j]]
            if not check_shape or sum(get_poster(movie_id).shape) == sum(config.POSTER_SHAPE):
                taken.append(movie_id)
            j += 1
        picked.append(pool.loc[taken])
    return pd.concat(picked)
