"""Plate scanner model wrapper.

We start from a pretrained Mask R-CNN (ResNet-50 FPN) and swap the class head
for `num_food_classes`. Training from scratch would be madness at v0 — we do
not have enough labeled plate data to justify it.

SAM can replace the segmentation stage later; that alternative lives in
`plate_sam.py` when we need it. For v0, Mask R-CNN is faster to deploy and
already gives usable masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PlateModelConfig:
    num_food_classes: int  # excludes background
    pretrained: bool = True
    # Image size is a training-time choice; export baked in at conversion time.
    image_size: int = 640


def build_plate_model(config: PlateModelConfig) -> nn.Module:
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if config.pretrained else None,
    )

    # num_classes includes background, so +1.
    num_classes = config.num_food_classes + 1

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def portion_grams_from_mask(
    mask_area_px: float,
    image_width_px: int,
    reference_plate_diameter_cm: float,
    reference_plate_diameter_px: float,
    density_g_per_cm3: float,
    *,
    assumed_thickness_cm: float = 1.5,
) -> float:
    """Estimate grams from a mask area.

    Uses a reference plate for scale (px -> cm) and a per-class density prior
    with a fixed assumed thickness. Replacing this with a learned estimator
    is a v1 task — the heuristic is deliberately simple so failures are
    easy to interpret during v0 review.
    """
    cm_per_px = reference_plate_diameter_cm / reference_plate_diameter_px
    mask_area_cm2 = mask_area_px * (cm_per_px**2)
    volume_cm3 = mask_area_cm2 * assumed_thickness_cm
    return volume_cm3 * density_g_per_cm3


def wrap_for_export(model: nn.Module, example_image: torch.Tensor) -> nn.Module:
    """Return a thin eval-mode wrapper that takes a single image tensor.

    Mask R-CNN's forward signature is a list of images -> list of dicts, which
    ONNX doesn't love. The wrapper unpacks to tensors so export is stable.
    """

    class PlateExportModule(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self, image: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.inner([image])[0]
            return (
                outputs["boxes"],
                outputs["labels"],
                outputs["scores"],
                outputs["masks"],
            )

    wrapped = PlateExportModule(model)
    wrapped.eval()
    # Trace once so the caller can trust forward() immediately.
    with torch.no_grad():
        wrapped(example_image)
    return wrapped
