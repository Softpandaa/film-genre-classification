"""EfficientNetV2 and Fast-ViT fine-tuning, section 5.

Ported as these notebooks ran, with two properties worth knowing before the
numbers are read. Both loaders wrap the whole frame: the notebooks computed a
train and test split and never passed it to the datasets, so the reported
metrics are measured on the rows the model trained on. And the training loop
applies a softmax before CrossEntropyLoss, which applies log-softmax again.

The population is read from encoded_genres.csv rather than rebuilt from the raw
Kaggle tables, so the rows line up with evaluate.py.
"""

import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm

import config


def population() -> pd.DataFrame:
    """Single-genre movies with their label, release year and table row."""
    table = pd.read_csv(config.ENCODED_GENRES_CSV)
    one_hot = table.iloc[:, 1:-1].to_numpy(bool)
    dates = pd.read_csv(config.MOVIES_CSV, index_col=0)[config.DATE]
    frame = pd.DataFrame({
        "row": np.arange(len(table)),
        config.MOVIE_ID: table[config.MOVIE_ID],
        "label": one_hot.argmax(axis=1),
        config.DATE: table[config.MOVIE_ID].map(dates),
    })
    return frame[(table[config.GENRE_COUNT] == 1).to_numpy()].dropna()


class PosterFrameDataset(Dataset):
    """Posters keyed by a frame of movie ids, labels and years."""

    def __init__(self, frame: pd.DataFrame, year_codes: dict):
        self.frame = frame.reset_index(drop=True)
        self.year_codes = year_codes
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(os.path.join(config.POSTER_DIR,
                                        f"{int(row[config.MOVIE_ID])}.jpg"))
        year = self.year_codes[row[config.DATE]]
        return self.transform(image), int(year), int(row["label"])


class EfficientNetWithYear(nn.Module):
    """EfficientNetV2-S with a frozen body and an optional year embedding."""

    def __init__(self, num_classes: int, num_years: int = 0):
        super().__init__()
        self.model = efficientnet_v2_s(weights="IMAGENET1K_V1")
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False
        features = self.model.classifier[-1].in_features
        self.num_years = num_years
        extra = num_years * config.YEAR_EMBEDDING_DIM
        if num_years:
            self.year_embedding = nn.Embedding(num_years,
                                               config.YEAR_EMBEDDING_DIM)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(features + extra, 1000, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1000, num_classes, bias=True),
        )

    def forward(self, x, years=None):
        features = self.model.avgpool(self.model.features(x))
        features = features.squeeze(-1).squeeze(-1)
        if self.num_years:
            one_hot = F.one_hot(years.long(), num_classes=self.num_years).float()
            embedded = self.year_embedding(one_hot.long())
            features = torch.cat((features, embedded.view(len(features), -1)), dim=1)
        return self.model.classifier(features)


def build_fastvit(num_classes: int):
    model = timm.create_model(config.FASTVIT_ARCH, pretrained=True,
                              num_classes=num_classes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("head")
    return model


def loaders(frame: pd.DataFrame, batch_size: int):
    """Train and validation loaders, both over the whole frame.

    This is what the notebooks built. They also computed

        train_idx, val_idx = train_test_split(frame.index, 0.2, random_state=42)

    and then passed the unsplit frame to both datasets, so the split had no
    effect. Kept as it ran.
    """
    years = {year: code for code, year in enumerate(sorted(frame[config.DATE].unique()))}
    return (DataLoader(PosterFrameDataset(frame, years), batch_size=batch_size,
                       shuffle=True, num_workers=4),
            DataLoader(PosterFrameDataset(frame, years), batch_size=batch_size,
                       shuffle=False, num_workers=4),
            len(years))


def train(model, train_loader, epochs: int, device: str = None,
          parameters=None) -> nn.Module:
    """Fine-tune the head. The softmax before the criterion is the notebooks'."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(parameters or model.parameters(), lr=config.TIMM_LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for images, years, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = (model(images, years.to(device))
                       if isinstance(model, EfficientNetWithYear) and model.num_years
                       else model(images))
            labels = F.one_hot(labels, num_classes=outputs.shape[1]).float()
            loss = criterion(F.softmax(outputs, dim=1), labels)
            loss.backward()
            optimizer.step()
    return model


def predict(model, loader, n_rows: int, frame: pd.DataFrame, name: str,
            device: str = None) -> str:
    """Write scores back into table order so evaluate.py can read them."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    scores = []
    with torch.no_grad():
        for images, years, _ in tqdm(loader, desc=name):
            outputs = (model(images.to(device), years.to(device))
                       if isinstance(model, EfficientNetWithYear) and model.num_years
                       else model(images.to(device)))
            scores.append(F.softmax(outputs, dim=1).cpu().numpy())
    scores = np.concatenate(scores)
    full = np.zeros((n_rows, scores.shape[1]))
    full[loader.dataset.frame["row"].to_numpy()] = scores
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
    np.savez_compressed(path, full)
    return path
