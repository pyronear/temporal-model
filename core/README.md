# temporal-model-core

Core building blocks of the bbox-tube temporal smoke classifier. Import as
`temporal_model.core`.

## Modules

- `protocol.py` — the `TemporalModel` contract (`Frame`, `TemporalModelOutput`,
  the `TemporalModel` ABC) and `parse_timestamp`. Vendored so the repo is
  self-contained.
- `types.py` — `Detection`, `FrameDetections`, `Tube`, `TubeEntry`.
- `tubes.py` — greedy-IoU tube linking, gap interpolation, colocated-tube merge.
- `crop.py` — pure bbox geometry (expand → pixel-square → crop/resize), shared by
  the inference and offline-training crop paths.
- `stabilize.py` — per-tube fixed crop window (union of observed boxes).
- `temporal_classifier.py` — `TimmBackbone` (ViT) + `TransformerHead` +
  `TemporalSmokeClassifier` (one logit per tube).
- `inference.py` — the per-stage inference pipeline (pad → YOLO → tubes → crop →
  score → first-crossing trigger).
- `model.py` — `BboxTubeTemporalModel`, the `TemporalModel` implementation.
- `package.py` — `model.zip` build/load (YOLO + classifier + calibrator + config)
  and `load_yolo`.
- `logistic_calibrator.py` — runtime logistic calibrator (pure numpy) and
  `tube_feature_dict`.
- `details_schema.py` — pydantic schema for `predict()` output details.
- `sequences.py`, `labels.py` — sequence discovery and detection/label/record loading.
- `detector.py`, `fetch_detector.py` — companion-detector identity + verified fetch.
- `stage_timer.py` — optional per-stage wall-clock profiling.

The classifier is **ViT-only** (transformer head on a timm ViT backbone, e.g.
`vit_small_patch14_dinov2.lvd142m`); the earlier mean-pool/GRU heads and
resnet/convnext backbones are intentionally not carried over.

```bash
make install
make test
```
