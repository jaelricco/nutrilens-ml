# nutrilens-ml

ML pipeline for NutriLens — plate scanning, pour volume estimation, and confidence-driven follow-up questions.

Primary deploy target: **CoreML on device**. Server-side **ONNX Runtime** exists as a fallback for heavier models and shadow evaluation.

See `docs/ROADMAP.md` for phase-by-phase build order.

## Layout (target, built over Phase 1)

```
nutrilens-ml/
├── docs/
│   └── ROADMAP.md
├── src/nutrilens_ml/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── export/
│   ├── serve/
│   └── eval/
├── bench/                # versioned held-out benchmarks
├── runs/                 # local training runs (git-ignored)
├── data/                 # cached datasets (git-ignored)
├── pyproject.toml
└── .env.example
```

## Getting started

Nothing to run yet — the repo is at Phase 0. Start at Phase 1 in `docs/ROADMAP.md`.
