"""VGG16 transfer learning, section 4.1.

Three genres, balanced, restricted to the release window in config.VGG16_YEARS,
with the genre count taken within the three rather than over all nineteen. The
sample is drawn as the notebook drew it, so a movie can be picked twice and the
train and test split below is made afterwards over those draws.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import config
from data_io import balanced_sample, get_poster, single_label_rows


def build(n_genre: int = 3):
    base = keras.applications.VGG16(weights="imagenet", include_top=False,
                                    input_shape=config.POSTER_SHAPE)
    base.trainable = False
    inputs = keras.Input(shape=config.POSTER_SHAPE)
    x = layers.Flatten()(base(inputs, training=False))
    x = layers.Dense(config.VGG16_DENSE_UNITS, activation="sigmoid")(x)
    x = layers.Dropout(config.VGG16_DROPOUT)(x)
    outputs = layers.Dense(n_genre, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.VGG16_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_sample(genres: list = None, years: tuple = config.VGG16_YEARS,
                strict: bool = False,
                n_per_genre: int = config.VGG16_SAMPLES_PER_GENRE,
                seed: int = config.VGG16_SEED):
    """Balanced posters and labels over the whole corpus.

    The release date is merged on the movie id. The notebook took it by row
    position, `movie_year.iloc[i - 1000001]`, which assumes the ids run
    contiguously from 1000001 in row order.
    """
    genres = genres or config.THREE_GENRES_TRANSFER
    rows = single_label_rows(genres=genres, strict=strict)
    if years is not None:
        dates = pd.read_csv(config.MOVIES_CSV, index_col=0)[config.DATE]
        year = rows.index.map(dates)
        rows = rows[(year >= years[0]) & (year <= years[1])]

    picked = balanced_sample(rows, genres, n_per_genre, seed)
    images = np.array([get_poster(i) for i in picked.index])
    labels = picked[[config.GENRE_PREFIX + g for g in genres]].to_numpy(float)
    return images, labels


def run(genres: list = None, years: tuple = config.VGG16_YEARS,
        strict: bool = False):
    """Fit and return the model with its held-out split."""
    genres = genres or config.THREE_GENRES_TRANSFER
    images, labels = load_sample(genres, years, strict)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=config.TEST_SIZE, random_state=config.VGG16_SEED)
    keras.utils.set_random_seed(config.VGG16_FIT_SEED)
    model = build(len(genres))
    model.fit(x_train, y_train, batch_size=config.BATCH_SIZE,
              epochs=config.VGG16_EPOCHS, validation_split=0.2)
    return model, (x_test, y_test)


# ---------------------------------------------------------------------------
# Model inspection, section 4.1. Each returns a transformed copy of x_test.
# ---------------------------------------------------------------------------

def monocolour(x_test: np.ndarray) -> np.ndarray:
    """Collapse to grey by averaging the channels."""
    grey = x_test.mean(axis=3)
    return np.repeat(grey[..., None], 3, axis=3)


def corners_removed(x_test: np.ndarray, pad: int = config.VGG16_CORNER_PAD):
    """Blank a border of `pad` pixels on every side."""
    out = np.zeros_like(x_test, dtype=float)
    out[:, pad:-pad, pad:-pad, :] = x_test[:, pad:-pad, pad:-pad, :]
    return out


def rotated(x_test: np.ndarray) -> np.ndarray:
    """Turn the poster upside down."""
    return np.rot90(x_test, k=2, axes=(1, 2))


def inspect(model, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Confusion matrix and macro F1 under each transformation."""
    truth = np.argmax(y_test, axis=1)
    results = {}
    for name, transform in [("original", lambda x: x), ("monocolour", monocolour),
                            ("corners removed", corners_removed),
                            ("rotated", rotated)]:
        predicted = np.argmax(model.predict(transform(x_test)), axis=1)
        results[name] = {"f1": f1_score(truth, predicted, average="macro"),
                         "confusion": confusion_matrix(truth, predicted)}
    return pd.DataFrame(results).T
