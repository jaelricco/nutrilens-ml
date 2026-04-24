"""Plate classifier training loop (ResNet-50 + CrossEntropy)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from nutrilens_ml.models.plate_classifier import build_plate_classifier
from nutrilens_ml.utils.seed import set_global_seed

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    num_classes: int
    epochs: int = 5
    batch_size: int = 64
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    run_dir: Path = Path("runs/plate-classifier/default")
    wandb: bool = False
    wandb_tags: list[str] = field(default_factory=list)
    class_names: list[str] | None = None


def _maybe_init_wandb(cfg: TrainConfig) -> Any | None:
    if not cfg.wandb:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb requested but `track` extra not installed; skipping")
        return None
    return wandb.init(project="nutrilens", tags=cfg.wandb_tags, config=asdict(cfg))


def train_classifier(
    train_ds: Dataset[Any], val_ds: Dataset[Any], cfg: TrainConfig
) -> Path:
    set_global_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.class_names is not None:
        (cfg.run_dir / "class_names.json").write_text(
            json.dumps(cfg.class_names, indent=2)
        )

    model = build_plate_classifier(num_classes=cfg.num_classes, pretrained=True).to(
        cfg.device
    )
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )

    run = _maybe_init_wandb(cfg)
    best_top1 = -1.0
    best_ckpt = cfg.run_dir / "best.pt"
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step, float(loss))
                if run is not None:
                    run.log({"train/loss": float(loss)}, step=step)
        sched.step()

        top1, top5 = _validate(model, val_loader, cfg.device)
        logger.info(
            "epoch=%d val/top1=%.4f val/top5=%.4f", epoch, top1, top5
        )
        if run is not None:
            run.log({"val/top1": top1, "val/top5": top5, "epoch": epoch}, step=step)

        if top1 > best_top1:
            best_top1 = top1
            cfg_dict = asdict(cfg)
            cfg_dict["run_dir"] = str(cfg_dict["run_dir"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg_dict,
                    "epoch": epoch,
                    "top1": top1,
                    "top5": top5,
                },
                best_ckpt,
            )

    if run is not None:
        run.finish()
    return best_ckpt


def _validate(model: nn.Module, loader: DataLoader[Any], device: str) -> tuple[float, float]:
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, preds5 = logits.topk(5, dim=-1)
            top1 += int((preds5[:, 0] == labels).sum())
            top5 += int((preds5 == labels.unsqueeze(-1)).any(dim=-1).sum())
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return top1 / total, top5 / total
