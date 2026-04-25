# Scorecard: plate-classifier 0.1.0

- bench: `food101-test-split` (not the frozen `bench/v0/` set — see notes)
- commit: `1199d20`
- generated: 2026-04-25
- artifact: `r2://nutrilens-ml/models/plate-classifier/0.1.0/`

## Metrics

| metric | value | v0 bar | status |
|--------|-------|--------|--------|
| top-1 accuracy | 0.8684 | — | — |
| top-5 accuracy | 0.9745 | ≥ 0.60 | **PASS** |

## Training setup

| field | value |
|-------|-------|
| model | ResNet-50 (ImageNet warm-start) |
| dataset | Food-101 (75,750 train / 25,250 test) |
| classes | 101 |
| epochs | 5 |
| batch size | 64 |
| optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| LR schedule | CosineAnnealingLR |
| loss | CrossEntropyLoss |
| seed | 42 |
| hardware | Colab Pro, T4 GPU |

## Notes

This is the **v0 warm-start** classifier, not the eventual plate detector. It satisfies the Phase 3 release criterion for top-5 accuracy (≥ 0.60), with significant headroom (0.97). top-1 of 0.87 is in line with published ResNet-50 + 5-epoch fine-tunes on Food-101.

**Caveats:**
- Evaluated on the Food-101 test split, not the frozen `bench/v0/` set. The benchmark suite was not yet wired into the Colab notebook for this release; subsequent releases should run `nutrilens_ml eval --bench v0` before publishing a scorecard.
- mIoU and grams MAE are not applicable — this is a classifier, not the detector. Those metrics only land once the Mask R-CNN trainer in `models/plate.py` runs against mask-labeled data.
- Calibration (temperature scaling per `followup/calibration.py`) was not yet applied. Should be fit before the on-device pipeline relies on `confidence` thresholds for follow-up question triggers.

**Next:**
1. Convert ONNX → CoreML on macOS (`nutrilens-ml export plate --checkpoint <onnx>`) and integrate `.mlpackage` into the iOS target.
2. Fit temperature scaling on the test logits and record ECE before / after.
3. Train and benchmark the actual detector (`models/plate.py`) once mask-labeled data is available; this classifier's weights serve as the backbone init.
