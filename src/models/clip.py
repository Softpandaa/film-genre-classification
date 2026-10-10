"""CLIP zero-shot classification of Section 5.1, equation (4)."""

import os

import clip
import numpy as np
import torch
from tqdm import tqdm

import config
from data_io import genre_names, load_encoded_genres
from models.timm_models import population

PROMPT = "A movie poster of {} genre"


def prompts(genres: list):
    return clip.tokenize([PROMPT.format(genre) for genre in genres])


def zero_shot(backbone: str, genres: list = None, device: str = None) -> np.ndarray:
    """Score every dated single-genre movie by image and text similarity."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(backbone, device=device)
    table = load_encoded_genres()
    genres = genres or genre_names(table.columns[:-1])
    text = prompts(genres).to(device)

    frame = population()
    scores = np.zeros((len(table), len(genres)))
    from PIL import Image
    with torch.no_grad():
        for _, row in tqdm(frame.iterrows(), total=len(frame), desc=backbone):
            path = os.path.join(config.POSTER_DIR,
                                f"{int(row[config.MOVIE_ID])}.jpg")
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            logits, _ = model(image, text)
            scores[int(row["row"])] = logits.softmax(dim=-1).cpu().numpy()
    return scores


def save(scores: np.ndarray, name: str) -> str:
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
    np.savez_compressed(path, scores)
    return path
