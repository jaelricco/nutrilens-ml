"""Food-101 loader.

Torchvision ships the downloader + splits. We only own the transform
choice. The ImageNet-style normalisation matches the pretrained ResNet
backbone; don't change it unless you're also retraining the backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_food101_datasets(
    root: Path, image_size: int = 224
) -> tuple[Any, Any, list[str]]:
    from torchvision import transforms as T
    from torchvision.datasets import Food101

    root.mkdir(parents=True, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = Food101(str(root), split="train", download=True, transform=train_tf)
    val_ds = Food101(str(root), split="test", download=True, transform=val_tf)

    class_names = [cls.replace("_", " ") for cls in train_ds.classes]
    return train_ds, val_ds, class_names
