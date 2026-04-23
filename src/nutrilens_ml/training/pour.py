"""Pour detector training loop.

Same shape as `training/plate.py`: plain PyTorch, one file, no Lightning.
Differences: the model wants a `liquid_idx` tensor and the loss uses both
the total-ml and per-frame-cumulative targets when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from nutrilens_ml.models.pour import PourLoss, PourModel, PourModelConfig
from nutrilens_ml.utils.seed import set_global_seed

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    run_dir: Path = Path("runs/pour/default")
    early_stop_patience_epochs: int = 6
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


def train_pour(
    train_ds: Dataset[Any],
    val_ds: Dataset[Any],
    cfg: TrainConfig,
) -> Path:
    set_global_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    model = PourModel(PourModelConfig()).to(cfg.device)
    loss_fn = PourLoss(aux_weight=cfg.aux_weight)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    run = _maybe_init_wandb(cfg)
    best_err = float("inf")
    best_ckpt = cfg.run_dir / "best.pt"
    epochs_since_improve = 0
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            clip = batch["clip"].to(cfg.device)
            liquid_idx = batch["liquid_idx"].to(cfg.device)
            total = batch["total_ml"].to(cfg.device)
            per_frame = batch.get("per_frame_cumulative_ml")
            if per_frame is not None:
                per_frame = per_frame.to(cfg.device)

            pred = model(clip, liquid_idx)
            loss = loss_fn(pred, total, per_frame)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if run is not None:
                run.log({"train/loss": float(loss)}, step=step)

        sched.step()

        median_err = _validate(model, val_loader, cfg.device)
        logger.info("epoch=%d val/median_abs_err_ml=%.3f", epoch, median_err)
        if run is not None:
            run.log({"val/median_abs_err_ml": median_err, "epoch": epoch}, step=step)

        if median_err < best_err:
            best_err = median_err
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_ckpt)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.early_stop_patience_epochs:
                logger.info("early stop at epoch %d (best median=%.3f)", epoch, best_err)
                break

    if run is not None:
        run.finish()
    return best_ckpt


def _validate(model: PourModel, loader: DataLoader[Any], device: str) -> float:
    model.eval()
    diffs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            clip = batch["clip"].to(device)
            liquid_idx = batch["liquid_idx"].to(device)
            total = batch["total_ml"].to(device)
            pred = model(clip, liquid_idx)["total_ml"]
            diffs.extend((pred - total).abs().tolist())
    if not diffs:
        return float("inf")
    diffs.sort()
    return diffs[len(diffs) // 2]
