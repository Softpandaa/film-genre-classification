"""The from-scratch CNN of section 3, with decade indicators.

Twelve convolutional layers on the raw (345, 230, 3) poster, with six decade
indicators concatenated to the flattened features. Trained on three genres at a
time, balanced, on the train half of the shared split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import config
from data_io import balanced_sample, get_poster, single_label_rows


def build(n_genre: int = 3, n_indicators: int = len(config.CNN_DECADES)):
    """Twelve convolutional layers, then the indicators, then two dense blocks."""
    body = keras.Sequential()
    body.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                           input_shape=config.POSTER_SHAPE))
    body.add(layers.GroupNormalization())
    for block, filters in enumerate([32, 32, 32, 64]):
        for layer in range(2 if block == 0 else 3):
            body.add(layers.Conv2D(filters, (3, 3), activation="relu",
                                   padding="same"))
            body.add(layers.GroupNormalization())
        body.add(layers.MaxPooling2D((2, 2)))
    body.add(layers.Flatten())

    indicators = keras.Input(shape=(n_indicators,))
    x = layers.Concatenate()([body.output, indicators])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(n_genre, activation="softmax")(x)

    model = keras.Model(inputs=[body.input, indicators], outputs=output)
    model.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def decade_indicators(years) -> np.ndarray:
    """One column per decade in config.CNN_DECADES."""
    years = np.asarray(years).reshape(-1)
    return np.column_stack([((years >= d) & (years < d + 10)).astype(int)
                            for d in config.CNN_DECADES])


def load_sample(genres: list, min_year: int = 1970, strict: bool = False,
                n_per_genre: int = config.CNN_SAMPLES_PER_GENRE,
                seed: int = config.CNN_SAMPLE_SEED):
    """Balanced posters, labels and release years, over the whole corpus."""
    rows = single_label_rows(genres=genres, strict=strict)
    dates = pd.read_csv(config.CNN_GENRES_CSV, index_col=0)[config.DATE]
    years = pd.to_datetime(dates).dt.year
    rows = rows[rows.index.map(years) > min_year]

    picked = balanced_sample(rows, genres, n_per_genre, seed)
    images = np.array([get_poster(i) for i in picked.index], dtype=float) / 255.0
    labels = picked[[config.GENRE_PREFIX + g for g in genres]].to_numpy(float)
    return images, labels, years.loc[picked.index].to_numpy()


def run(genres: list, min_year: int = 1970, strict: bool = False):
    """Fit one three-genre setting and return the model and its test split."""
    images, labels, years = load_sample(genres, min_year, strict)
    x_train, x_test, y_train, y_test, year_train, year_test = train_test_split(
        images, labels, years, test_size=config.TEST_SIZE,
        random_state=config.CNN_SPLIT_SEED)
    model = build(n_genre=len(genres))
    model.fit([x_train, decade_indicators(year_train)], y_train,
              batch_size=config.BATCH_SIZE, epochs=config.EPOCHS,
              validation_split=0.2)
    return model, (x_test, y_test, year_test)
