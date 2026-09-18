"""How the frozen tables in data/ were built, and a check that they are intact.

The four tables named in config are the corpus of record. Kaggle has revised
the Letterboxd dataset since the project ran, so these builders no longer
reproduce them, and they refuse to overwrite an existing file for that reason.
They are kept because they document the construction, and because verify_corpus
is what catches a table that has drifted.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from data_io import get_poster


def _guard(path: str, overwrite: bool) -> None:
    """Refuse to replace a frozen table unless asked explicitly. Checked first,
    before any of the raw tables are read."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} is part of the frozen corpus. Rebuilding from the current "
            "Kaggle download produces a different table. Pass overwrite=True "
            "only if you intend to replace the corpus of record.")


def _write(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    frame.to_csv(path)
    return frame


def verify_corpus() -> pd.DataFrame:
    """Row count and index hash of each frozen table against config."""
    paths = {"encoded_genres": config.ENCODED_GENRES_CSV,
             "valid_movies": config.VALID_MOVIES_CSV,
             "poster_sizes": config.POSTER_SIZES_CSV,
             "cnn_genres": config.CNN_GENRES_CSV}
    rows = []
    for name, path in paths.items():
        frame = pd.read_csv(path, index_col=0)
        digest = hashlib.md5(frame.index.to_numpy().tobytes()).hexdigest()
        expected_rows, expected_digest = config.CORPUS_FINGERPRINT[name]
        rows.append({"table": name, "rows": len(frame),
                     "ok": len(frame) == expected_rows and digest == expected_digest})
    return pd.DataFrame(rows)


def build_poster_sizes(overwrite: bool = False) -> pd.DataFrame:
    """Shape of every readable poster. Unreadable files are skipped."""
    _guard(config.POSTER_SIZES_CSV, overwrite)
    ids = pd.read_csv(config.GENRES_CSV)[config.ID].unique()
    ids.sort()
    sizes = {}
    for movie_id in tqdm(ids, desc="poster sizes"):
        try:
            sizes[movie_id] = get_poster(movie_id).shape
        except Exception:
            continue
    table = pd.DataFrame({config.MOVIE_ID: list(sizes.keys()),
                          config.SIZE: list(sizes.values())})
    return _write(table, config.POSTER_SIZES_CSV)


def build_valid_movies(overwrite: bool = False) -> pd.Series:
    """Movies whose poster has the modal shape."""
    _guard(config.VALID_MOVIES_CSV, overwrite)
    sizes = pd.read_csv(config.POSTER_SIZES_CSV, index_col=0)
    modal = sizes[config.SIZE].mode()
    valid = sizes[sizes[config.SIZE].isin(modal)][config.MOVIE_ID]
    _write(valid.to_frame(), config.VALID_MOVIES_CSV)
    return valid


def build_encoded_genres(overwrite: bool = False) -> pd.DataFrame:
    """One-hot genre table over valid movies, outliers in genre count removed.

    A movie is an outlier when its genre count exceeds the mean plus three
    standard deviations, which leaves at most four genres per movie.
    """
    _guard(config.ENCODED_GENRES_CSV, overwrite)
    genres = pd.read_csv(config.GENRES_CSV, index_col=0)
    counts = genres.groupby(config.ID).count()
    upper = counts[config.GENRE].mean() + 3 * counts[config.GENRE].std()
    kept = counts[counts[config.GENRE] < upper]

    valid = pd.read_csv(config.VALID_MOVIES_CSV, index_col=0)
    cleaned = sorted(set(valid[config.MOVIE_ID]).intersection(set(kept.index)))

    table = genres[genres.index.isin(cleaned)]
    encoded = pd.get_dummies(table, columns=[config.GENRE],
                             prefix="").groupby(config.ID).any()
    encoded[config.GENRE_COUNT] = encoded.sum(axis=1)
    encoded.index.names = [config.MOVIE_ID]
    return _write(encoded, config.ENCODED_GENRES_CSV)


def build_cnn_genres(overwrite: bool = False) -> pd.DataFrame:
    """The scratch CNN table: earliest release date, after 1970, up to three genres."""
    _guard(config.CNN_GENRES_CSV, overwrite)
    releases = pd.read_csv(config.RELEASES_CSV, index_col=0)
    releases[config.DATE] = pd.to_datetime(releases[config.DATE])
    earliest = releases.groupby(config.ID)[config.DATE].agg("min")

    encoded = pd.read_csv(config.ENCODED_GENRES_CSV, index_col=0)
    merged = pd.merge(encoded, earliest, left_index=True, right_index=True,
                      how="left").dropna()
    merged = merged[merged[config.DATE].dt.year > 1970]
    merged = merged[merged[config.GENRE_COUNT] <= 3]
    return _write(merged, config.CNN_GENRES_CSV)


def sample_decade_posters(decades=("1950s", "1960s", "1970s", "1980s", "1990s",
                                   "2000s", "2010s", "2020s"),
                          n: int = 10, seed: int = 4563, country: str = "Hong Kong",
                          min_rating: float = 3.0) -> dict:
    """Sample posters per decade for the appendix figures.

    The pool is Hong Kong releases, and from 1980 onward only those rated at or
    above min_rating. The notebook this came from had only the 2010s left in
    its decade list when it was saved, and sampled five, while the committed
    figures show ten per decade, so n follows the figures rather than the
    notebook and only the 2010s ids are verified against them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = pd.read_csv(config.VALID_MOVIES_CSV, index_col=0)
    movies = pd.read_csv(config.MOVIES_CSV, index_col=0)
    countries = pd.read_csv(config.COUNTRIES_CSV, index_col=0)

    dated = valid.merge(movies[[config.DATE, config.RATING]], left_on=config.MOVIE_ID,
                        right_on=config.ID, how="left").dropna()
    dated = dated.query("1950 <= date <= 2024")
    dated = dated.merge(countries, left_on=config.MOVIE_ID, right_on=config.ID,
                        how="left").dropna()
    dated = dated[dated[config.COUNTRY] == country]

    os.makedirs(config.FIG_DIR, exist_ok=True)
    np.random.seed(seed)
    sampled = {}
    for decade in decades:
        start = int(decade[:4])
        pool = dated.query(f"{start} <= date <= {start + 9}")
        if start >= 1980:
            pool = pool[pool[config.RATING] >= min_rating]
        sampled[decade] = sorted(int(i) for i in pool[config.MOVIE_ID].sample(n=n))

        figure, axes = plt.subplots(1, n, figsize=(2.0 * n, 3.0))
        for axis, movie_id in zip(axes, sampled[decade]):
            axis.imshow(get_poster(movie_id))
            axis.set_title(str(movie_id), fontsize=9)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(os.path.join(config.FIG_DIR, f"posters_{decade}.png"),
                       bbox_inches="tight", dpi=150)
        plt.close(figure)
    return sampled


if __name__ == "__main__":
    print(verify_corpus().to_string(index=False))
