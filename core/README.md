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
- `pipeline.py` — `TubePipelineModel`, the backend-agnostic `predict()` (torch-free).
- `model.py` — `BboxTubeTemporalModel`, the torch + YOLO backend.
- `onnx_model.py` — `OnnxTemporalModel`, the onnxruntime backend for `model_onnx.zip`
  (supplied detections only, no torch on its import path).
- `export_onnx.py` — `temporal-export-onnx`: derive `model_onnx.zip` from a `model.zip`
  and verify torch/ONNX parity.
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

## Installing

The base package is torch-free (numpy, pillow, pydantic, pyyaml). Pick an extra:

- `temporal-model-core[torch]` — training, packaging, `BboxTubeTemporalModel`,
  the ONNX export (torch, timm, ultralytics, onnx, onnxscript).
- `temporal-model-core[onnx]` — `OnnxTemporalModel` on `model_onnx.zip`
  (onnxruntime only; edge devices).

```bash
make install   # dev environment: both extras
make test
```

## Running the model

Both backends run the same pipeline and produce the same logits (parity is
checked at export, `atol=1e-4`); pick one by machine, not by feature.

### ONNX, torch-free (edge devices)

The ONNX backend bundles no YOLO: supply the per-frame detections from the
detector already running on the device, boxes as normalized `(cx, cy, w, h)`.

```bash
pip install "temporal-model-core[onnx] @ git+https://github.com/pyronear/temporal-model.git#subdirectory=core"
```

```python
from pathlib import Path

from huggingface_hub import hf_hub_download
from temporal_model.core import Detection, FrameDetections
from temporal_model.core.onnx_model import OnnxTemporalModel

onnx_zip = hf_hub_download("pyronear/temporal-model", "model_onnx.zip", revision="v0.4.0")
model = OnnxTemporalModel.from_package(Path(onnx_zip))

# Temporally ordered frames (filename order is time order).
frames = model.load_sequence(sorted(Path("my_sequence").glob("*.jpg")))

# One FrameDetections per frame, keyed by frame_id.
detections = {
    f.frame_id: FrameDetections(
        frame_idx=i, frame_id=f.frame_id, timestamp=f.timestamp,
        detections=[Detection(class_id=0, cx=0.5, cy=0.4, w=0.05, h=0.03, confidence=0.6)],
    )
    for i, f in enumerate(frames)
}

out = model.predict(frames, frame_detections=detections)
print("is_smoke:", out.is_positive)
kept = out.details["tubes"]["kept"]
probs = [t["probability"] for t in kept if t["probability"] is not None]
print("smoke probability:", max(probs) if probs else 0.0)
```

To run the ONNX session on a GPU, install `onnxruntime-gpu` and pass
`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` to
`from_package`.

### Torch + GPU (bundled YOLO)

The torch backend runs its bundled YOLO itself — pass only the images.
`device=None` auto-selects `cuda` → `mps` → `cpu`.

```bash
pip install "temporal-model-core[torch] @ git+https://github.com/pyronear/temporal-model.git#subdirectory=core"
```

```python
from pathlib import Path

from huggingface_hub import hf_hub_download
from temporal_model.core.model import BboxTubeTemporalModel

model_zip = hf_hub_download("pyronear/temporal-model", "model.zip", revision="v0.4.0")
model = BboxTubeTemporalModel.from_package(Path(model_zip), device="cuda")

out = model.predict_sequence(sorted(Path("my_sequence").glob("*.jpg")))
print("is_smoke:", out.is_positive)
kept = out.details["tubes"]["kept"]
probs = [t["probability"] for t in kept if t["probability"] is not None]
print("smoke probability:", max(probs) if probs else 0.0)
```
