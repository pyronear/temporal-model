---
license: apache-2.0
library_name: pytorch
tags:
  - wildfire
  - smoke-detection
  - object-detection
  - temporal
---

# Temporal Smoke Model (bbox-tube-temporal)

> **Latest release:** [`v{{VERSION}}`](https://huggingface.co/pyronear/temporal-model/tree/v{{VERSION}}) — pin this revision for reproducibility, or omit `revision=` to always get the latest. All releases: the **Files and versions** tab.

A temporal wildfire-**smoke** classifier for short sequences of camera frames. A
YOLO detector proposes boxes, boxes are linked across frames into temporal
**tubes**, each tube's image patches are classified by a DINOv2 ViT + transformer
head, and a logistic calibrator turns the tube logits into a calibrated
probability and a keep/discard decision.

This repo ships a single self-contained **`model.zip`**, versioned by HuggingFace
revision/tag (`v<version>`). Each release bundles everything needed to run:

| file | purpose |
|---|---|
| `manifest.yaml` | version + provenance (train git SHA, backbone, detector) |
| `yolo_weights.pt` | the companion YOLO detector |
| `classifier.ckpt` | the temporal ViT classifier |
| `config.yaml` | inference + decision config |
| `logistic_calibrator.json` | the calibrated decision head |

The model runs YOLO **itself** — you pass only raw frames, no detections.

## Usage

Install the inference package (`temporal_model.core`):

```bash
pip install "git+https://github.com/pyronear/temporal-model.git#subdirectory=core"
```

Download a versioned `model.zip` and run it on a **temporally ordered** sequence
of frames:

```python
from pathlib import Path

from huggingface_hub import hf_hub_download
from temporal_model.core.model import BboxTubeTemporalModel

# 1. Download a specific release (pin the revision).
model_zip = hf_hub_download("pyronear/temporal-model", "model.zip", revision="v{{VERSION}}")

# 2. Temporally-ordered frames. Filenames carry timestamps
#    (<prefix>_<YYYY-MM-DDTHH-MM-SS>.jpg); the order is the time order.
frame_paths = sorted(Path("my_sequence").glob("*.jpg"))

# 3. Load (device=None → auto cuda → mps → cpu) and predict.
#    hf_hub_download returns a str, so wrap it in Path().
model = BboxTubeTemporalModel.from_package(Path(model_zip), device=None)
out = model.predict_sequence(frame_paths)

# 1. Binary verdict — the alarm decision.
print("is_smoke:         ", out.is_positive)

# 2. Calibrated smoke probability in [0, 1]: the strongest kept tube's calibrated
#    probability (0.0 when no tube was kept). Use is_positive for a yes/no alarm;
#    use the probability to rank/triage sequences or apply your own threshold.
kept = out.details.get("tubes", {}).get("kept", [])
probs = [t["probability"] for t in kept if t["probability"] is not None]
smoke_probability = max(probs) if probs else 0.0
print("smoke probability:", smoke_probability)
print("kept tubes:       ", len(kept))
```

`predict_sequence(frame_paths)` returns a `TemporalModelOutput`:

- `is_positive: bool` — the smoke verdict. `True` iff at least one tube's
  calibrated probability clears the packaged decision threshold.
- `details: dict` — per-tube logits, calibrated probabilities, bounding boxes, and
  the decision (`aggregation`, `threshold`). The top-level **smoke probability** is
  the maximum kept-tube `probability` (shown above).

`is_positive` and the probability are consistent: a positive sequence always has a
kept tube whose probability is ≥ the threshold, so `smoke_probability` is ≥ the
threshold whenever `is_smoke` is `True`.

## Served API (Docker)

The same model is also served as a FastAPI image with the `model.zip` baked in
(auto-uses the GPU with `--gpus all`):

```bash
docker run --gpus all -p 8000:8000 \
  -e TEMPORAL_API_S3_BUCKET=<frames-bucket> \
  -e TEMPORAL_API_S3_ENDPOINT_URL=<s3-endpoint> \
  pyronear/temporal-model-api:{{VERSION}}
# POST /predict  {"frames": ["<s3-key>", ...]}      GET /health
```

`POST /predict` returns `{ "is_smoke": bool, "probability": float | null, "model":
{...} }`, where `probability` is the calibrated max kept-tube probability (`null`
only for an uncalibrated model). Add `?verbose=true` for a `details` block with the
per-tube breakdown and decision config.

## Provenance

Every `model.zip` manifest records how it was built — the training git SHA, the
classifier backbone (`vit_small_patch14_dinov2.lvd142m`), and the exact companion
detector (e.g. `pyronear/yolo11s_nimble-narwhal_v6.0.0`, verified by SHA-256). So
a served model always traces back to its detector + training code.

Source & pipeline: <https://github.com/pyronear/temporal-model>
