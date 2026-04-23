"""Dataset schemas.

Every sample carries a stable `sample_id` so splits and content-hash caches
stay deterministic across runs. `schema_version` is bumped on breaking changes
so old manifests fail loudly instead of silently misaligning.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat

SCHEMA_VERSION = 1


class LiquidType(StrEnum):
    olive_oil = "olive_oil"
    cream = "cream"
    milk = "milk"
    soy_sauce = "soy_sauce"
    liquid_chocolate = "liquid_chocolate"
    vinegar = "vinegar"


class FoodLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    # FDC (FoodData Central) ID or an internal stable ID when FDC is missing.
    fdc_id: int | None = None
    name: str


class PlateItem(BaseModel):
    label: FoodLabel
    mask_path: Path | None = None  # relative to dataset root; optional for classification-only samples
    grams: NonNegativeFloat
    bbox_xywh: tuple[float, float, float, float] | None = None


class PlateSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    sample_id: str
    image_path: Path
    items: list[PlateItem]
    reference_plate_diameter_cm: PositiveFloat | None = None


class PourSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    sample_id: str
    video_path: Path
    liquid: LiquidType
    total_ml: NonNegativeFloat
    # Optional per-frame cumulative ml, used as auxiliary loss in Phase 4.
    per_frame_cumulative_ml: list[NonNegativeFloat] | None = None
    fps: PositiveFloat = 30.0
    container_id: str | None = None


class DatasetManifest(BaseModel):
    """Top-level manifest for a versioned dataset snapshot."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    name: str
    version: str
    root: Path
    plate: list[PlateSample] = Field(default_factory=list)
    pour: list[PourSample] = Field(default_factory=list)
