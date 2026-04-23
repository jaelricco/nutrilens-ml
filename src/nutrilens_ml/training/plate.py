"""Plate scanner training loop.

Kept deliberately compact: one file, plain PyTorch, no Lightning abstraction.
The shape is dataloader -> forward -> loss -> step -> (periodic) val -> ckpt.
W&B logging is optional (requires the `track` extra); we fall back to stdout.

This is the scaffold — the dataloader adapter lives in `data/plate_loader.py`
(added when real labeled data lands).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from nutrilens_ml.models.plate import PlateModelConfig, build_plate_model
from nutrilens_ml.utils.seed import set_global_seed

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    num_food_classes: int
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    run_dir: Path = Path("runs/plate/default")
    early_stop_patience_epochs: int = 5
    wandb: bool = False
    wandb_tags: list[str] = field(default_factory=list)


def _maybe_init_wandb(cfg: TrainConfig) -> Any | None:
    if not cfg.wandb:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb requested but `track` extra not installed; skipping")
        return None
    return wandb.init(project="nutrilens", tags=cfg.wandb_tags, config=cfg.__dict__)


def _log(run: Any | None, payload: dict[str, float], step: int) -> None:
    logger.info("step=%d %s", step, payload)
    if run is not None:
        run.log(payload, step=step)


def train_plate(
    train_ds: Dataset[Any],
    val_ds: Dataset[Any],
    cfg: TrainConfig,
) -> Path:
    """Run training and return the path of the best-F1 checkpoint."""
    set_global_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    model = build_plate_model(PlateModelConfig(num_food_classes=cfg.num_food_classes))
    model.to(cfg.device)

    # Mask R-CNN returns losses in train mode; that's what we backprop.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate)

    run = _maybe_init_wandb(cfg)
    best_f1 = -1.0
    best_ckpt = cfg.run_dir / "best.pt"
    epochs_since_improve = 0
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for images, targets in train_loader:
            images = [img.to(cfg.device) for img in images]
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            _log(run, {"train/loss": float(loss)}, step)
        sched.step()

        f1 = _validate(model, val_loader, cfg.device)
        _log(run, {"val/f1": f1, "epoch": epoch}, step)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_ckpt)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.early_stop_patience_epochs:
                logger.info("early stop at epoch %d (best val/f1=%.4f)", epoch, best_f1)
                break

    if run is not None:
        run.finish()
    return best_ckpt


def _collate(
    batch: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=False)
    return list(images), list(targets)


def _validate(model: nn.Module, loader: DataLoader[Any], device: str) -> float:
    # Real metric computation lives in `eval/plate.py`; here we just need a
    # scalar to drive early stopping. Compute macro-F1 over predicted labels.
    from nutrilens_ml.eval.plate import macro_f1_over_loader

    model.eval()
    with torch.no_grad():
        return macro_f1_over_loader(model, loader, device)
