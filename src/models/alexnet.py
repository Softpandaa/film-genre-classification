"""AlexNet with a multilabel head, section 4.2."""

import torch.nn as nn
from torch import optim
from torchvision import models

import config
from models.lightning import PretrainedModel, head

# Output of AlexNet's convolutional stack, flattened.
FEATURES = 9216


def build(lr: float = None, optimizer=None):
    backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return PretrainedModel(backbone, head(FEATURES), config.N_GENRES,
                           loss_fn=nn.BCELoss,
                           optimizer=optimizer or optim.Adam,
                           lr=lr or config.TORCHVISION_LR["alexnet"])
