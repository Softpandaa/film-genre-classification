"""ResNet-18 and ResNet-50 with a multilabel head, section 4.2."""

import torch.nn as nn
from torch import optim
from torchvision import models

import config
from models.lightning import PretrainedModel, head

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
}


def build(name: str = "resnet18", lr: float = None, optimizer=None,
          **optimizer_params):
    constructor, weights = BACKBONES[name]
    backbone = constructor(weights=weights)
    return PretrainedModel(backbone, head(backbone.fc.in_features), config.N_GENRES,
                           loss_fn=nn.BCELoss,
                           optimizer=optimizer or optim.Adam,
                           lr=lr or config.TORCHVISION_LR[name],
                           **optimizer_params)
