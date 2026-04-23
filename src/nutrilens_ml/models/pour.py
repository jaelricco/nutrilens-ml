"""Pour detector model.

Backbone: torchvision MViT-v2-S as the scaffold default. It's overkill for
on-device but beats the alternatives in torchvision for accuracy. When we
hit the latency bar we'll swap to X3D-XS via pytorchvideo — that's a
deliberate v1 task, not a v0 blocker.

Conditioning: a learned embedding for `LiquidType` is concatenated with the
pooled backbone feature before the regression head. One-hot would work but
the embedding lets the model share across similar viscosities (cream/milk)
instead of treating every liquid as orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nutrilens_ml.data.schemas import LiquidType

NUM_LIQUIDS = len(LiquidType.__members__)


@dataclass(frozen=True)
class PourModelConfig:
    clip_frames: int = 64
    frame_size: int = 224
    liquid_embedding_dim: int = 16
    hidden_dim: int = 256
    pretrained: bool = True
    aux_head_enabled: bool = True  # per-frame cumulative ml; disabled at inference


class PourModel(nn.Module):
    def __init__(self, cfg: PourModelConfig) -> None:
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        weights = MViT_V2_S_Weights.DEFAULT if cfg.pretrained else None
        backbone = mvit_v2_s(weights=weights)
        # Strip the classification head — we want the pooled feature.
        feature_dim = backbone.head[-1].in_features
        backbone.head = nn.Identity()
        self.backbone = backbone

        self.liquid_embedding = nn.Embedding(NUM_LIQUIDS, cfg.liquid_embedding_dim)

        fused_dim = feature_dim + cfg.liquid_embedding_dim
        self.total_head = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.aux_head_enabled = cfg.aux_head_enabled
        if cfg.aux_head_enabled:
            self.per_frame_head = nn.Sequential(
                nn.Linear(fused_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.clip_frames),
            )

    def forward(
        self, clip: torch.Tensor, liquid_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # clip: (B, C, T, H, W)
        feat = self.backbone(clip)  # (B, D)
        liq = self.liquid_embedding(liquid_idx)  # (B, E)
        fused = torch.cat([feat, liq], dim=-1)
        out: dict[str, torch.Tensor] = {"total_ml": self.total_head(fused).squeeze(-1)}
        if self.aux_head_enabled and self.training:
            out["per_frame_cumulative_ml"] = self.per_frame_head(fused)
        return out


def liquid_to_index(liquid: LiquidType) -> int:
    return list(LiquidType.__members__).index(liquid.name)


class PourLoss(nn.Module):
    """L1(total) + aux_weight * L1(per-frame cumulative).

    No mixup on the liquid embedding — the minority-class liquids need their
    own gradients, and mixing hurts them empirically.
    """

    def __init__(self, aux_weight: float = 0.3) -> None:
        super().__init__()
        self.aux_weight = aux_weight

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target_total: torch.Tensor,
        target_per_frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.nn.functional.l1_loss(pred["total_ml"], target_total)
        if (
            self.aux_weight > 0
            and "per_frame_cumulative_ml" in pred
            and target_per_frame is not None
        ):
            loss = loss + self.aux_weight * torch.nn.functional.l1_loss(
                pred["per_frame_cumulative_ml"], target_per_frame
            )
        return loss
