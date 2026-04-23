"""FastAPI inference server.

The iOS app never talks to this service — only the Rust backend does. Inputs
are presigned URLs into the same R2 / S3 bucket the backend uploads to, so
we never proxy raw bytes through Rust.

Routes:
- GET /healthz    — liveness
- GET /readyz     — readiness (models loaded, registry reachable)
- POST /infer/plate
- POST /infer/pour
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from nutrilens_ml.data.schemas import LiquidType
from nutrilens_ml.followup.rules import generate_questions
from nutrilens_ml.serve.auth import require_shared_secret

logger = logging.getLogger(__name__)

app = FastAPI(title="NutriLens ML", version="0.0.1", docs_url=None, redoc_url=None)


class PlateRequest(BaseModel):
    image_url: HttpUrl
    model_version: str = Field(default="latest")


class PourRequest(BaseModel):
    video_url: HttpUrl
    liquid: LiquidType
    model_version: str = Field(default="latest")


class InferenceItem(BaseModel):
    label: str
    fdc_id: int | None = None
    grams: float | None = None
    kcal: float | None = None
    volume_ml: float | None = None
    confidence: float


class InferenceResponse(BaseModel):
    schema_version: int = 1
    task: str
    items: list[InferenceItem]
    overall_confidence: float
    followup_questions: list[str] = Field(default_factory=list)
    model_version: str
    task_extras: dict[str, object] = Field(default_factory=dict)


@app.get("/healthz", summary="liveness")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", summary="readiness")
def readyz() -> dict[str, bool]:
    # Phase 7 will make this actually check the model registry.
    plate_ready = Path(os.getenv("PLATE_MODEL_PATH", "")).is_file()
    pour_ready = Path(os.getenv("POUR_MODEL_PATH", "")).is_file()
    if not (plate_ready and pour_ready):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"plate": plate_ready, "pour": pour_ready},
        )
    return {"plate": plate_ready, "pour": pour_ready}


@app.post(
    "/infer/plate",
    response_model=InferenceResponse,
    dependencies=[Depends(require_shared_secret)],
)
def infer_plate(req: PlateRequest) -> InferenceResponse:
    # Phase 7 wires this to the real pipeline: fetch image -> preprocess ->
    # plate_infer() -> postprocess -> calibrate confidences -> generate_questions.
    stub_items = [
        InferenceItem(label="pasta", grams=220, kcal=310, confidence=0.74),
    ]
    prediction = {
        "items": [i.model_dump() for i in stub_items],
        "overall_confidence": 0.74,
    }
    return InferenceResponse(
        task="plate",
        items=stub_items,
        overall_confidence=0.74,
        followup_questions=generate_questions(prediction),
        model_version=req.model_version,
    )


@app.post(
    "/infer/pour",
    response_model=InferenceResponse,
    dependencies=[Depends(require_shared_secret)],
)
def infer_pour(req: PourRequest) -> InferenceResponse:
    # Phase 7 wires this to: fetch video -> sample frames -> pour_infer() ->
    # apply per-ml kcal from pantry product -> respond.
    estimated_ml = 12.5
    item = InferenceItem(
        label=req.liquid.value,
        volume_ml=estimated_ml,
        confidence=0.72,
    )
    return InferenceResponse(
        task="pour",
        items=[item],
        overall_confidence=0.72,
        followup_questions=[],
        model_version=req.model_version,
    )
