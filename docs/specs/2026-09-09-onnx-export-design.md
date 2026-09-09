# ONNX Export and Torch-Free Runtime

- **PR:** [#76 — feat: ONNX export and torch-free runtime for the tube classifier](https://github.com/pyronear/temporal-model/pull/76)
- **Date:** 2026-09-09
- **Status:** Implemented

## Problem

Edge devices (Raspberry Pi class) already run their own YOLO detector and
cannot afford the torch + timm + ultralytics stack (multi-GB install, slow
cold start) just to run the tube classifier. The tube pipeline itself —
pad → detect → tubes → crop → score → decide — is numpy work except for two
stages: detection and tube scoring.

## Decisions

### Pipeline split

`predict()` moves out of `BboxTubeTemporalModel` into a backend-agnostic
`TubePipelineModel` (`core/.../pipeline.py`), which must stay importable
without torch (enforced by `test_onnx_runtime_import_never_loads_torch`).
Backends provide two hooks:

- `_score(patches[N,T,3,P,P] float32, mask[N,T] bool) -> logits[N]`
- `detect(frames)` — optional; backends without a bundled detector raise
  `NotImplementedError`, and their `predict()` requires `frame_detections`
  covering every frame.

`BboxTubeTemporalModel` (torch + YOLO) and `OnnxTemporalModel` (onnxruntime,
supplied detections only) are the two backends. Preprocessing
(`crop_tube_patches`) is rewritten in pure numpy, bit-equivalent to the old
torchvision `to_tensor` path — guarded by `test_model_parity.py`.

### Artifact: `model_onnx.zip`

Derived from a released `model.zip` by `temporal-export-onnx`
(`core/.../export_onnx.py`), published next to it under the same HF tag.
Contents:

- `manifest.yaml` — `format_version`, file pointers, source `model.zip`
  identity (name + SHA-256), and the ONNX I/O contract (names, shapes,
  dtypes, opset, `max_abs_logit_diff` from the parity check).
- `classifier.onnx` — self-contained (no external data), so
  `load_onnx_package` can build the session from bytes without extracting.
- `config.yaml` / `logistic_calibrator.json` — copied verbatim from the source.

The source SHA-256 pairs the two artifacts: `release publish --onnx-file`
refuses an ONNX archive whose manifest was not exported from that exact
`model.zip`, then re-stamps the hash from the staged (version-stamped) copy
it actually uploads.

### I/O contract: dynamic batch, fixed frames

`patches[batch, max_frames, 3, P, P]` + `mask[batch, max_frames]` →
`logit[batch]`. The batch axis is dynamic so `_score` is a single
`session.run` for all tubes (matching the torch backend's batched forward);
`max_frames` and `patch_size` are fixed at export from the package config —
shorter tubes are always padded to `max_frames` by the pipeline anyway, so a
dynamic time axis would buy nothing and cost tracing risk.

### Parity check at export

`verify_export` compares torch vs onnxruntime logits on random inputs over
four mask patterns (all real, padded tail, single frame, interleaved gaps),
each at batch 1 and once all together as one batch, with `atol=1e-4` on
logits. The torch reference is a **deep copy in its as-served configuration**
(fused attention untouched); only the exported copy gets `fused_attn=False`
(timm's fused kernels don't trace portably). `export_classifier` and
`verify_export` never mutate the caller's module.

### Calibration gate

`build_onnx_package` refuses an uncalibrated `model.zip` by default
(`--allow-uncalibrated` to override), mirroring
`2026-06-09-enforce-model-calibration-design.md`. Refusing at export keeps an
uncalibrated artifact from being published under an immutable tag and only
failing on the edge device at `load_onnx_package` time.

### Dependency split

Core's base package is numpy + pillow + pydantic + pyyaml. Extras:

- `[torch]` — torch backend, training/packaging, and the ONNX export
  (which needs onnx + onnxscript to export and onnxruntime to verify).
- `[onnx]` — onnxruntime only: the edge runtime (161 MB install on a Pi 5).

`UncalibratedModelError` and the shared helpers live in `pipeline.py` (torch-
free) and are re-exported from both `package.py` (torch path) and
`onnx_model.py` (torch-free path). The `requires-python < 3.13` cap is owned
by `numpy < 2` (no 3.13 wheels), not by the export stack.

## Alternatives considered

- **Bundling YOLO in the ONNX package** — rejected: edge devices already run
  their own detector; shipping a second one doubles the artifact for nothing.
- **TorchScript instead of ONNX** — rejected: still requires torch on the
  device, which is the dependency being removed.
- **Extracting the archive on load** — rejected: the session is built from
  in-memory bytes; no temp-file churn, no cache directory on the device.
