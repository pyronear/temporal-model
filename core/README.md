# temporal-model-core

Core building blocks of the bbox-tube temporal smoke classifier. Import as
`temporal_model.core`.

## Modules

- `protocol.py` — the `TemporalModel` contract (`Frame`, `TemporalModelOutput`,
  the `TemporalModel` ABC). Vendored so the repo is self-contained.
- `types.py` — `Detection`, `FrameDetections`, `Tube`, `TubeEntry`, ….
- `tubes.py` — greedy-IoU tube linking, gap interpolation, colocated-tube merge.
- `model_input.py` — bbox expansion + 224×224 patch cropping.
- `temporal_classifier.py` — `TimmBackbone` (ViT) + `TransformerHead` +
  `TemporalSmokeClassifier` (one logit per tube).
- `inference.py` — the per-stage inference pipeline (pad → YOLO → tubes → crop →
  score → first-crossing trigger).
- `model.py` — `BboxTubeTemporalModel`, the `TemporalModel` implementation.
- `package.py` — `model.zip` build/load (YOLO + classifier + calibrator + config).
- `logistic_calibrator.py` — runtime logistic calibrator (pure numpy).
- `details_schema.py`, `data.py` — output schema and detection/frame loading.

The classifier is **ViT-only** (transformer head on a timm ViT backbone, e.g.
`vit_small_patch14_dinov2.lvd142m`); the earlier mean-pool/GRU heads and
resnet/convnext backbones are intentionally not carried over.

```bash
make install
make test
```
