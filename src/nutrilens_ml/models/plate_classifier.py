"""Plate classifier — ResNet-50 fine-tuned on Food-101.

This is the v0 *warm-start* for plate recognition. The detector (Mask R-CNN
in `plate.py`) wants segmentation masks, which we don't have yet. A
classifier uses labels we already have access to via Food-101, so we can
actually train something real today. When mask-labeled data lands, we
swap the detector in and reuse the weights here as a backbone init.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class PlateClassifierConfig:
    num_classes: int
    pretrained: bool = True


def build_plate_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
