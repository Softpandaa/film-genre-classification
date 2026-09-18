"""YOLOv8 classification, section 4.3.

Two approaches. Multi-label copies a poster into one folder per genre it
carries. Multi-class gives each movie its single genre. Both build their folder
dataset from the train and test halves of the shared split, so the validation
folder is the evaluation set.
"""

import os
import shutil

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import config
from data_io import load_encoded_genres, genre_names, split_indices


def build_folders(root: str, multi_label: bool = True) -> str:
    """Lay posters out as <root>/<split>/<genre>/<movie_id>.jpg."""
    table = load_encoded_genres().reset_index()
    names = genre_names(table.columns[1:-1])
    train_indices, _ = split_indices(len(table))
    in_train = np.isin(np.arange(len(table)), train_indices)

    for split in ("train", "val"):
        for genre in names:
            os.makedirs(os.path.join(root, split, genre), exist_ok=True)

    one_hot = table.iloc[:, 1:-1].to_numpy(bool)
    for row, movie_id in enumerate(tqdm(table[config.MOVIE_ID], desc="posters")):
        carried = np.flatnonzero(one_hot[row])
        if not multi_label:
            if len(carried) != 1:
                continue
            carried = carried[:1]
        split = "train" if in_train[row] else "val"
        for column in carried:
            shutil.copy(os.path.join(config.POSTER_DIR, f"{movie_id}.jpg"),
                        os.path.join(root, split, names[column], f"{movie_id}.jpg"))
    return root


def train(root: str, settings: dict, weights: str = config.YOLO_WEIGHTS):
    model = YOLO(weights)
    model.train(data=root, batch=config.BATCH_SIZE, optimizer="Adam",
                workers=config.NUM_WORKERS, exist_ok=True, save=True, **settings)
    return model


def predict_all(model, name: str) -> str:
    """Score every row of the table, in table order, for evaluate.py."""
    table = load_encoded_genres().reset_index()
    scores = [model(os.path.join(config.POSTER_DIR, f"{movie_id}.jpg"),
                    verbose=False)[0].probs.data.cpu().numpy()
              for movie_id in tqdm(table[config.MOVIE_ID], desc=name)]
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
    np.savez_compressed(path, np.array(scores))
    return path
