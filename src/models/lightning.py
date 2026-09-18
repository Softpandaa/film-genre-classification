"""The Lightning wrapper and the training loop shared by AlexNet and the ResNets.

Kept as the project ran it: a Sigmoid head with BCELoss over the nineteen
genres, Adam, five epochs, batch 32.
"""

import os

import lightning as L
import numpy as np
import torch.nn as nn
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader

import config
from dataset import PosterDataset, get_datasets


class PretrainedModel(L.LightningModule):
    """A frozen-body backbone with a trainable multilabel head."""

    def __init__(self, pre_trained_model, last_layer_classifier, num_labels: int,
                 loss_fn=nn.BCELoss, optimizer=optim.Adam, **optimizer_params):
        super().__init__()
        self.features = nn.Sequential(*list(pre_trained_model.children())[:-1])
        self.class_classifier = last_layer_classifier
        self.loss_fn = loss_fn()
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.metrics = torchmetrics.classification.MultilabelAUROC(num_labels)

    def forward(self, x):
        return self.class_classifier(self.features(x).view(x.size(0), -1))

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_params)

    def _step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat.view(y.size()), y)
        self.log(f"{stage}_auroc", self.metrics(y_hat, y.long()), prog_bar=True,
                 on_step=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)


def head(in_features: int, num_labels: int = config.N_GENRES):
    return nn.Sequential(nn.Linear(in_features, num_labels), nn.Sigmoid())


def train(model: PretrainedModel, epochs: int = config.EPOCHS, logger=None,
          root_dir: str = None):
    """Fit on the train half of the shared split, validate on the test half."""
    train_dataset, test_dataset = get_datasets()
    loaders = [
        DataLoader(d, batch_size=config.BATCH_SIZE, shuffle=s,
                   num_workers=config.NUM_WORKERS, pin_memory=True)
        for d, s in ((train_dataset, True), (test_dataset, False))
    ]
    trainer = L.Trainer(max_epochs=epochs, logger=logger,
                        default_root_dir=root_dir or config.ARTIFACT_DIR)
    trainer.fit(model, *loaders)
    return model


def predict_all(model: PretrainedModel, name: str) -> str:
    """Score every row of the table and save the array evaluate.py expects."""
    loader = DataLoader(PosterDataset(), batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    predictions = np.concatenate(L.Trainer().predict(model, loader))
    os.makedirs(config.ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACT_DIR, f"{name}_predictions.npz")
    np.savez_compressed(path, predictions)
    return path
