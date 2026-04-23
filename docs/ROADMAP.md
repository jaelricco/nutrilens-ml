# Development Roadmap — ML Pipeline Build Order

## Current Focus

We are building the **NutriLens ML pipeline** from scratch. Follow this order strictly — each phase produces an artifact the next phase consumes.

The pipeline has three model families:
1. **Plate scanner** — segment + identify + estimate portions of food on a plate (image → items + grams + kcal).
2. **Pour detector** — estimate liquid volume from a short video, conditioned on the selected pantry product's liquid type.
3. **Confidence + follow-up** — calibrated uncertainty that triggers hidden-ingredient prompts when overall confidence < 80%.

Primary deployment target is **CoreML on device** (latency, privacy, cost). An **ONNX Runtime server** exists as a fallback for models too heavy for A-series silicon and for A/B shadow evaluation.

## Phase 1: Project Scaffolding

### Step 1.1: Initialize the project
- `uv init` (or `poetry init`) — pinned Python 3.11
- `src/` layout: `nutrilens_ml/{data,models,training,export,serve,eval}`
- `pyproject.toml` with core deps: `torch`, `torchvision`, `onnx`, `onnxruntime`, `coremltools`, `pydantic`, `pydantic-settings`, `typer`, `pytest`, `ruff`, `mypy`
- `.env.example` with dataset bucket, experiment tracker keys, model registry path
- `Dockerfile` for the server (CPU base; GPU variant deferred to Phase 6)

### Step 1.2: Configuration module
- `nutrilens_ml/config.py` — `pydantic-settings` loaded from env + optional YAML override
- No secrets in code, no `assert` for validation — raise `ConfigError` with a clear message
- Single `Settings` object imported wherever config is read

### Step 1.3: Experiment tracking & reproducibility
- Pick one: **Weights & Biases** (managed) or **MLflow** (self-host). Default: W&B.
- Every training run logs: config hash, dataset hash, git SHA, seed, metrics, final weights
- Deterministic seed helper: `set_global_seed(seed)` sets `torch`, `numpy`, `random`, `PYTHONHASHSEED`

### Step 1.4: CLI entry point
- `python -m nutrilens_ml <command>` via `typer`
- Commands: `train`, `eval`, `export`, `serve`, `ingest`
- Every command takes `--config <path>` and `--run-name <str>`

## Phase 2: Data Pipeline

### Step 2.1: Dataset schemas
- `nutrilens_ml/data/schemas.py` — `pydantic` models for each dataset type
- **Plate dataset**: image path, segmentation masks, food labels (FDC / USDA IDs), per-item grams, reference object (plate diameter in cm)
- **Pour dataset**: video path, liquid type (enum from pantry taxonomy), final volume ml, per-frame volume (optional, used for aux loss)
- Labels versioned — breaking changes bump `schema_version`

### Step 2.2: Ingestion from object storage
- Pull raw data from the same R2 / S3 bucket the backend uploads to
- Local cache with content-hashed filenames under `data/cache/` (git-ignored)
- Deterministic train/val/test split by stable hash of sample ID, never by shuffle+index

### Step 2.3: Bootstrap datasets
- **Plate**: start with **Food-101** + **UECFood-256** for classification; synthesize portion labels with a plate-diameter reference until real labeled data arrives
- **Pour**: collect ~50 videos per liquid type as v0 (team + contractors); augment with rendered / simulated pours where viscosity curves are known
- Keep a README per dataset with license, source, collection protocol

### Step 2.4: Data QA
- `ingest` command runs sanity checks: corrupt files, label/image mismatch, class imbalance, mask coverage
- Failing samples go to `data/quarantine/` with a reason file — never silently dropped

## Phase 3: Plate Scanner v0

### Step 3.1: Architecture
- Segmentation: start from a pretrained **SAM** or **Mask R-CNN** backbone — do not train from scratch
- Classification head on top, fine-tuned on Food-101 + custom labels
- Portion estimator: pixel area × reference-object scale → grams, using per-class density priors. Neural estimator deferred to v1.

### Step 3.2: Training loop
- `nutrilens_ml/training/plate.py` — standard PyTorch `Trainer` pattern
- Logs to W&B, checkpoints every N steps to `runs/<run-name>/ckpt/`
- Early stopping on val F1, not val loss

### Step 3.3: Evaluation
- `nutrilens_ml/eval/plate.py` — top-1 / top-5 food accuracy, mIoU for segmentation, MAE for grams
- **Release bar for v0**: top-5 ≥ 60%, mIoU ≥ 0.55, MAE grams ≤ 30% of label
- Report per-class so we see where the long tail hurts

### Step 3.4: Export
- PyTorch → ONNX (opset 17, dynamic batch, fixed input size)
- ONNX → CoreML via `coremltools` with `ComputeUnit.ALL`
- Export smoke test: run a fixed image through both ONNX Runtime and the `.mlpackage` and assert output deltas < 1e-3

## Phase 4: Pour Detector v0

### Step 4.1: Architecture
- Input: 64-frame clip at 30 fps, 224×224, with liquid-type one-hot conditioning
- Backbone: **MoViNet-A0** or **X3D-XS** (mobile-first, already trained on Kinetics)
- Head: scalar regression (total ml) + optional per-frame ml head for auxiliary loss

### Step 4.2: Collection protocol
- Documented rig: phone on tripod, plain background, mass scale under the container, 30 fps, timestamps synced
- Liquid types from pantry taxonomy (olive oil, cream, milk, soy sauce, liquid chocolate, vinegar as v0)
- Each session logs: liquid ID, container geometry, camera distance, start/end mass

### Step 4.3: Training loop
- `nutrilens_ml/training/pour.py`
- Loss: `L1(volume) + 0.3 * L1(per-frame cumulative)` — auxiliary head stabilizes training
- Mixup on liquid embedding disabled — it hurts generalization on minority liquids

### Step 4.4: Evaluation
- **Release bar for v0**: median absolute error ≤ 15% of true ml, 90th percentile ≤ 35%
- Error broken down per liquid type — ship only liquids meeting the bar
- Latency budget on iPhone 15: ≤ 250 ms per 2-second clip

### Step 4.5: Export
- Same PyTorch → ONNX → CoreML path as plate
- Validate on-device latency on a real iPhone, not just the simulator

## Phase 5: Confidence & Follow-up Logic

### Step 5.1: Calibration
- Temperature scaling on val set for the plate classifier
- Record ECE (expected calibration error) before and after — must improve, else don't ship calibration

### Step 5.2: Follow-up question engine
- Rule table mapping `(food_label, missing_confidence_signal) → question`
- Examples: "Is there butter or oil in this?" when visible dish is a sauté; "What dressing?" when segmentation finds a salad with no dressing mask
- Lives in `nutrilens_ml/followup/rules.py` — plain Python, trivially testable. Model-based follow-ups deferred.

### Step 5.3: Schema for iOS
- Shared JSON schema in `docs/schemas/` for inference responses:
  ```json
  {
    "items": [{"label": "...", "grams": 42, "kcal": 120, "confidence": 0.83}],
    "overall_confidence": 0.71,
    "followup_questions": ["..."]
  }
  ```
- Backend re-emits this shape; iOS consumes it directly. No duplication of logic.

## Phase 6: Server-side Inference Service

### Step 6.1: FastAPI service
- `nutrilens_ml/serve/app.py` — FastAPI + `onnxruntime`
- Routes: `POST /infer/plate`, `POST /infer/pour`, `GET /healthz`, `GET /readyz`
- Input is a presigned-URL reference to a photo/video in R2 / S3 — service pulls, infers, returns JSON. Never proxy raw bytes through the Rust backend.

### Step 6.2: Auth
- Shared secret header from the Rust backend (not the iOS client)
- mTLS deferred until we deploy to production

### Step 6.3: Deployment
- Dockerfile with `onnxruntime` (CPU); separate `Dockerfile.gpu` with CUDA variant
- Deploy target: Fly.io CPU machine alongside the Rust backend for v0
- Auto-scale to zero out of dev-hours; cold start is fine for async flows

## Phase 7: Evaluation & Monitoring

### Step 7.1: Held-out benchmarks
- Frozen benchmark sets per task, versioned under `bench/`
- Every release runs `nutrilens_ml eval --bench v1` and publishes a scorecard to `docs/scorecards/`

### Step 7.2: Drift detection
- Log production inputs (hashed) and model outputs via the server
- Weekly job compares output-distribution shift vs. the benchmark — alert when KL divergence exceeds a threshold

### Step 7.3: Model registry
- Simple: an S3 / R2 prefix `models/<task>/<semver>/` with a `manifest.json` (metrics, dataset hash, training SHA)
- Rust backend reads `LATEST_PLATE_MODEL` / `LATEST_POUR_MODEL` env vars; promotion is a config change, not a deploy

## Phase 8: CI / CD & Release

### Step 8.1: Lint and test
- `ruff`, `mypy --strict`, `pytest` — all required to pass on PR
- Unit tests cover: data schema validation, config loading, follow-up rules, export parity (ONNX vs CoreML)

### Step 8.2: GitHub Actions
- On PR: lint + tests + an export smoke test on a tiny fixture model
- On merge to `main`: rebuild the server image, push to registry
- On tag `plate-vX.Y.Z` / `pour-vX.Y.Z`: run full eval, publish `.mlpackage` + ONNX + manifest to model registry

### Step 8.3: Release artifacts
- Per release, publish: `plate.mlpackage`, `plate.onnx`, `pour.mlpackage`, `pour.onnx`, scorecard, changelog
- iOS pulls `.mlpackage` at app build time from the model registry pinned by semver — never "latest"

---

## What NOT to Build Yet

These are real but premature. Do not build them during Phases 1–8:

- **Seasoning / spice estimation** — separately listed in the product roadmap; needs its own dataset, not a bolt-on to the plate model
- **Micronutrient prediction** — Pro+ feature, depends on label OCR data we don't yet have at scale
- **On-device training / personalization** — privacy-friendly in theory, expensive in practice, no evidence users need it yet
- **End-to-end differentiable volume pipeline** — separate segmentation → volume heuristic is good enough for v0 and much easier to debug
- **Model-based follow-up questions** — ship the rule table first; LLM-generated follow-ups only if rules are clearly insufficient
- **Real-time server inference from the iOS app** — all live inference is on device; server inference exists only behind the Rust backend

## Release Bars at a Glance

| Task | Metric | v0 Bar |
|------|--------|--------|
| Plate classification | top-5 accuracy | ≥ 60% |
| Plate segmentation | mIoU | ≥ 0.55 |
| Portion estimation | MAE as % of true grams | ≤ 30% |
| Pour detection (per liquid) | median abs error as % ml | ≤ 15% |
| Pour detection (per liquid) | p90 abs error as % ml | ≤ 35% |
| On-device latency (plate) | iPhone 15, end-to-end | ≤ 500 ms |
| On-device latency (pour) | iPhone 15, 2s clip | ≤ 250 ms |
