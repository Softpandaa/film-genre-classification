"""The poster dataset and the split shared by every PyTorch model."""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

import config
from data_io import get_poster, split_indices


class PosterDataset(Dataset):
    """Posters paired with their one-hot genre vector."""

    def __init__(self, indices=None, transform_=None):
        self.data = pd.read_csv(config.ENCODED_GENRES_CSV)
        if indices is not None:
            self.data = self.data.iloc[indices, :]
        self.transform = transform_ if transform_ is not None else default_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        movie_id, *features, _genre_count = self.data.iloc[idx, :]
        sample = self.transform(get_poster(movie_id))
        return sample, np.array(features).astype(np.float32)


def default_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(list(config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def get_datasets(**kwargs):
    """Train and test datasets under the shared split."""
    n = len(pd.read_csv(config.ENCODED_GENRES_CSV))
    train_indices, test_indices = split_indices(n)
    return PosterDataset(indices=train_indices, **kwargs), \
        PosterDataset(indices=test_indices, **kwargs)
