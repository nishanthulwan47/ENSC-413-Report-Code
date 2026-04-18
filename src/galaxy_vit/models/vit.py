from __future__ import annotations
import timm
import torch.nn as nn


class GalaxyViT(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x):
        return self.backbone(x)
