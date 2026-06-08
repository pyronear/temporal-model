# Stabilized Crops in Production Train + Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make stabilized per-tube crops (fixed union window) the production crop mode for the single `vit_dinov2_finetune` variant, threaded through `core`, produced by `train`, and reported by `eval` — with `stabilize` defaulting to `true` everywhere.

**Architecture:** `stabilize` is a crop-window decision. A new pure `core/stabilize.py::union_window` computes a tube's enclosing box; `process_tube` (offline/training crops) and `crop_tube_patches` (online/inference crops) take a `stabilize: bool = True` flag that crops that fixed window on every frame instead of each frame's own box. `model.predict` reads `stabilize` from the baked config; `package.py` bakes it from the `model_input.stabilize` param; `build_model_input.py` passes it via a `--stabilize` CLI arg wired from DVC. Eval is unchanged behaviourally — it crops from the baked `model.zip` config.

**Tech Stack:** Python 3.11+, PyTorch, PIL/numpy, pytest, DVC, uv, ruff. Monorepo with three editable packages: `core/`, `train/`, `eval/`.

---

## Background for the implementer (read once)

- The repo is a **single-variant** monorepo. `core/` holds the model library; `train/` builds + packages `data/06_models/vit_dinov2_finetune/model.zip`; `eval/` runs the protocol on that `model.zip`.
- `core/` mirrors the upstream experiment lib `vision-rd/lib/bbox-tube-temporal` but carries repo-specific additions. **Do not copy files wholesale** — apply only the edits in this plan.
- A tube is a `Tube` with `.entries: list[TubeEntry]`; each `TubeEntry` has `.frame_idx`, `.is_gap`, and `.detection` (a `Detection` with `.cx .cy .w .h`, normalized, or `None` for a gap). In the training-data JSON form, an entry is a dict with `"frame_idx"`, `"frame_id"`, `"bbox": [cx, cy, w, h]`, `"is_gap"`.
- `union_window` takes normalized `(cx, cy, w, h)` boxes and returns the normalized enclosing box. No context margin — the crop step adds context via `context_factor`.
- Run package commands from the package root. Per-package: `make test` (pytest), `make lint` (ruff check), `make format` (ruff format).
- Commit messages: imperative, scoped (e.g. `feat(core): ...`). **No Claude co-author trailer.**

---

## File map

| File | Change |
|---|---|
| `core/src/temporal_model/core/stabilize.py` | **Create** — `union_window` |
| `core/tests/test_stabilize.py` | **Create** — union_window tests |
| `core/src/temporal_model/core/model_input.py` | Add `stabilize` to `process_tube` |
| `core/tests/test_model_input.py` | Stabilize tests for `process_tube` |
| `core/src/temporal_model/core/inference.py` | Add `stabilize` to `crop_tube_patches` |
| `core/tests/test_inference_units.py` | Stabilize tests for `crop_tube_patches` |
| `core/src/temporal_model/core/model.py` | `predict` passes `stabilize=mi.get("stabilize", True)` |
| `core/tests/test_model_parity.py` | Parametrize parity over `stabilize` ∈ {False, True} |
| `train/src/temporal_model/train/build_model_input.py` | `--stabilize` CLI arg + thread through |
| `train/tests/test_build_model_input.py` | **Create** — `_to_bool` + `_process_one` stabilize |
| `train/src/temporal_model/train/package.py` | Bake `stabilize` into `build_config` |
| `train/tests/test_package_builders.py` | Assert baked `stabilize` |
| `train/params.yaml` | `model_input.stabilize: true` |
| `train/dvc.yaml` | Pass `--stabilize`, add `stabilize.py` deps |
| `eval/dvc.yaml` | Add `stabilize.py` dep |

---

## Task 1: `core/stabilize.py` — pure union-window helper

**Files:**
- Create: `core/src/temporal_model/core/stabilize.py`
- Test: `core/tests/test_stabilize.py`

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_stabilize.py`:

```python
"""Unit tests for the pure stable-crop-window helper."""

import pytest

from temporal_model.core.stabilize import union_window


def test_union_of_two_boxes_is_axis_independent():
    # A: x[0.15,0.25] y[0.45,0.55]; B: x[0.35,0.45] y[0.45,0.55]
    # union: x[0.15,0.45] (w=0.3, cx=0.3); y[0.45,0.55] (h=0.1, cy=0.5)
    boxes = [(0.2, 0.5, 0.1, 0.1), (0.4, 0.5, 0.1, 0.1)]
    assert union_window(boxes) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_single_box_returns_itself():
    assert union_window([(0.5, 0.5, 0.2, 0.2)]) == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_empty_raises():
    with pytest.raises(ValueError):
        union_window([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_stabilize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'temporal_model.core.stabilize'`

- [ ] **Step 3: Write minimal implementation**

Create `core/src/temporal_model/core/stabilize.py`:

```python
"""Per-tube fixed crop window (pure geometry, no I/O).

Stabilization is a crop-window decision, not a tube-building step. ``union_window``
returns the enclosing box of a tube's observed detection boxes so the same region
can be cropped from every frame — a static background with the smoke moving inside
it. No context margin here: the crop step adds context via ``context_factor``.
"""

from __future__ import annotations


def union_window(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Enclosing box of ``boxes`` (each normalized ``(cx, cy, w, h)``).

    Returns normalized ``(cx, cy, w, h)``. Raises ``ValueError`` if ``boxes`` is
    empty. Width and height are computed independently from the x/y extents.
    """
    if not boxes:
        raise ValueError("union_window requires at least one box")
    x0 = min(cx - w / 2 for cx, _, w, _ in boxes)
    y0 = min(cy - h / 2 for _, cy, _, h in boxes)
    x1 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y1 = max(cy + h / 2 for _, cy, _, h in boxes)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_stabilize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/stabilize.py core/tests/test_stabilize.py
git commit -m "feat(core): add union_window for stabilized crops"
```

---

## Task 2: `process_tube` stabilize (offline/training crops)

**Files:**
- Modify: `core/src/temporal_model/core/model_input.py`
- Test: `core/tests/test_model_input.py`

The existing `process_tube` crops each frame's own `bbox`. Add a `stabilize` flag (default `True`) that crops a single fixed window for every frame. `orig_bbox` in meta still records each entry's own box (unchanged). `meta.json` gains a `stabilize` field.

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_model_input.py` (the helpers `_write_jpg`, `_write_tube_record` already exist in this file):

```python
def _write_tube_record_2boxes(path: Path, sequence_id: str, frame_ids: list[str]):
    """Tube whose two frames have boxes at different x positions, so the union
    window (cx=0.5) differs from either frame's own box (cx=0.3 / cx=0.7)."""
    boxes = [[0.3, 0.5, 0.1, 0.1], [0.7, 0.5, 0.1, 0.1]]
    record = {
        "sequence_id": sequence_id,
        "split": "train",
        "label": "smoke",
        "source": "gt",
        "num_frames": len(frame_ids),
        "tube": {
            "start_frame": 0,
            "end_frame": len(frame_ids) - 1,
            "entries": [
                {
                    "frame_idx": i,
                    "frame_id": fid,
                    "bbox": boxes[i],
                    "is_gap": False,
                    "confidence": 0.9,
                }
                for i, fid in enumerate(frame_ids)
            ],
        },
    }
    path.write_text(json.dumps(record))
    return record


def test_process_tube_stabilize_uses_constant_window(tmp_path):
    seq_id = "site_999_2023-07-01T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=True,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    # Same fixed crop window pixels for every frame.
    crop_boxes = [f["crop_bbox_pixels"] for f in meta["frames"]]
    assert crop_boxes[0] == crop_boxes[1]
    # orig_bbox still records each frame's own detection box.
    assert meta["frames"][0]["orig_bbox"] == [0.3, 0.5, 0.1, 0.1]
    assert meta["frames"][1]["orig_bbox"] == [0.7, 0.5, 0.1, 0.1]
    assert meta["stabilize"] is True


def test_process_tube_default_is_stabilized(tmp_path):
    seq_id = "site_999_2023-07-02T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(  # no stabilize arg -> defaults to True
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is True
    assert meta["frames"][0]["crop_bbox_pixels"] == meta["frames"][1]["crop_bbox_pixels"]


def test_process_tube_stabilize_false_is_per_frame(tmp_path):
    seq_id = "site_999_2023-07-03T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=False,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is False
    # Different per-frame boxes -> different crop windows.
    assert meta["frames"][0]["crop_bbox_pixels"] != meta["frames"][1]["crop_bbox_pixels"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_model_input.py -k stabilize -v`
Expected: FAIL — `process_tube() got an unexpected keyword argument 'stabilize'`

- [ ] **Step 3: Implement**

In `core/src/temporal_model/core/model_input.py`, add the import near the other relative imports (after `from .data import find_sequence_dir`):

```python
from .stabilize import union_window
```

Replace the `process_tube` signature and body. Current signature ends `patch_size: int,` then `) -> None:`. New version:

```python
def process_tube(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
    stabilize: bool = True,
) -> None:
    record = json.loads(tube_path.read_text())
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        raise FileNotFoundError(f"raw sequence dir not found for {sequence_id}")

    images_dir = seq_dir / "images"
    seq_out = out_dir / sequence_id
    seq_out.mkdir(parents=True, exist_ok=True)

    entries = record["tube"]["entries"]
    window = None
    if stabilize:
        observed = [tuple(e["bbox"]) for e in entries if not e["is_gap"]]
        window = union_window(observed or [tuple(e["bbox"]) for e in entries])

    frame_meta: list[dict] = []
    for entry in entries:
        frame_id = entry["frame_id"]
        frame_idx = entry["frame_idx"]
        bbox = window if stabilize else entry["bbox"]
        is_gap = entry["is_gap"]

        img_path = images_dir / f"{frame_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], context_factor)
        crop_box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch = crop_and_resize(image, crop_box, patch_size)

        filename = f"frame_{frame_idx:02d}.png"
        save_patch(patch, seq_out / filename)

        frame_meta.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "is_gap": is_gap,
                "orig_bbox": list(entry["bbox"]),
                "crop_bbox_pixels": list(crop_box),
                "filename": filename,
            }
        )

    meta = {
        "sequence_id": sequence_id,
        "split": record["split"],
        "label": label,
        "label_int": LABEL_TO_INT[label],
        "num_frames": record["num_frames"],
        "context_factor": context_factor,
        "patch_size": patch_size,
        "stabilize": stabilize,
        "frames": frame_meta,
    }
    (seq_out / "meta.json").write_text(json.dumps(meta, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd core && uv run pytest tests/test_model_input.py -v`
Expected: PASS (all existing tests + the 3 new stabilize tests)

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/model_input.py core/tests/test_model_input.py
git commit -m "feat(core): stabilize crop window in process_tube (default true)"
```

---

## Task 3: `crop_tube_patches` stabilize (online/inference crops)

**Files:**
- Modify: `core/src/temporal_model/core/inference.py`
- Test: `core/tests/test_inference_units.py`

- [ ] **Step 1: Write the failing test**

`red_image_sequence` (the existing fixture) produces solid-red JPGs, so two *different* crop windows would still yield identical red patches — useless for distinguishing per-frame vs stabilized. Add a module-level **gradient** fixture (x-varying pixels) near `red_image_sequence` in `core/tests/test_inference_units.py`:

```python
@pytest.fixture()
def gradient_image_sequence(tmp_path: Path) -> list[Path]:
    """Three 128x128 horizontal-gradient JPGs (x-varying), so different crop
    windows produce different patches. All three frames are identical, so a
    fixed (stabilized) window crops identically across frames."""
    paths = []
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    img = np.stack([ramp, ramp, ramp], axis=-1)
    for i in range(3):
        p = tmp_path / f"frame_{i:02d}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=95)
        paths.append(p)
    return paths
```

Append two tests to the `TestCropTubePatches` class (the `_tube` and `_det` helpers already exist):

```python
    def test_stabilize_uses_constant_union_window(
        self, gradient_image_sequence: list[Path]
    ) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in gradient_image_sequence
        ]
        # Two frames, boxes at different x; union window is centered between them.
        tube = _tube(
            0,
            [
                (0, _det(cx=0.3, cy=0.5, w=0.1, h=0.1)),
                (1, _det(cx=0.7, cy=0.5, w=0.1, h=0.1)),
            ],
        )
        stab, _ = crop_tube_patches(
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
            stabilize=True,
        )
        per_frame, _ = crop_tube_patches(
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
            stabilize=False,
        )
        # Stabilized: both real frames share the same fixed window -> identical patches.
        assert torch.equal(stab[0], stab[1])
        # Per-frame: different boxes -> different patches.
        assert not torch.equal(per_frame[0], per_frame[1])

    def test_default_is_stabilized(self, gradient_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in gradient_image_sequence
        ]
        tube = _tube(
            0,
            [
                (0, _det(cx=0.3, cy=0.5, w=0.1, h=0.1)),
                (1, _det(cx=0.7, cy=0.5, w=0.1, h=0.1)),
            ],
        )
        default, _ = crop_tube_patches(  # no stabilize arg -> defaults to True
            tube, frames, context_factor=1.5, patch_size=224, max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert torch.equal(default[0], default[1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_inference_units.py -k "stabiliz or default_is_stabilized" -v`
Expected: FAIL — `crop_tube_patches() got an unexpected keyword argument 'stabilize'`

- [ ] **Step 3: Implement**

In `core/src/temporal_model/core/inference.py`, add to the imports (after line 16 `from .model_input import ...`):

```python
from .stabilize import union_window
```

Add `stabilize: bool = True,` to the `crop_tube_patches` keyword-only signature (after `normalization_std: list[float],`). Then, after the `std_t = ...` line and before the `for slot, entry in enumerate(...)` loop, insert:

```python
    window = None
    if stabilize:
        observed = [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in tube.entries
            if e.detection is not None and not e.is_gap
        ]
        if not observed:
            observed = [
                (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                for e in tube.entries
                if e.detection is not None
            ]
        window = union_window(observed)
```

Inside the loop, replace the per-frame expand line:

```python
        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, context_factor)
```

with:

```python
        box_src = window if stabilize else (det.cx, det.cy, det.w, det.h)
        cx, cy, w, h = expand_bbox(
            box_src[0], box_src[1], box_src[2], box_src[3], context_factor
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_inference_units.py -v`
Expected: PASS (all existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/inference.py core/tests/test_inference_units.py
git commit -m "feat(core): stabilize crop window in crop_tube_patches (default true)"
```

---

## Task 4: `model.predict` reads stabilize + parity test covers both modes

**Files:**
- Modify: `core/src/temporal_model/core/model.py`
- Test: `core/tests/test_model_parity.py`

`predict` must pass `stabilize` from the baked config to `crop_tube_patches`. The default is `True`, so the existing per-frame parity test (whose config has no `stabilize` key and whose offline reference crops per-frame) would break unless updated. We parametrize parity over both modes and make the offline reference stabilize-aware.

- [ ] **Step 1: Update the parity test (will fail until model.py changes)**

In `core/tests/test_model_parity.py`:

Add imports — `copy` at the top with the stdlib imports, and `union_window`:

```python
import copy
```
```python
from temporal_model.core.stabilize import union_window
```

In `_offline_logit_with_cfg`, make the crop loop stabilize-aware. Replace the loop body region that currently reads:

```python
    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, mi["context_factor"])
```

with:

```python
    window = None
    if mi.get("stabilize", True):
        observed = [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in tube.entries
            if e.detection is not None and not e.is_gap
        ]
        if not observed:
            observed = [
                (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                for e in tube.entries
                if e.detection is not None
            ]
        window = union_window(observed)

    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        box_src = window if mi.get("stabilize", True) else (det.cx, det.cy, det.w, det.h)
        cx, cy, w, h = expand_bbox(
            box_src[0], box_src[1], box_src[2], box_src[3], mi["context_factor"]
        )
```

Parametrize the parity test. Replace the `test_parity_logit_matches_transformer` definition header and its first/config lines:

```python
@pytest.mark.parametrize("stabilize", [False, True])
def test_parity_logit_matches_transformer(
    transformer_classifier: TemporalSmokeClassifier, stabilize: bool
) -> None:
    cfg = copy.deepcopy(CFG_TRANSFORMER)
    cfg["model_input"]["stabilize"] = stabilize
    offline = _offline_logit_with_cfg(transformer_classifier, cfg)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    model = BboxTubeTemporalModel(
        yolo_model=yolo,
        classifier=transformer_classifier,
        config=cfg,
        device="cpu",
    )
    out = model.predict(frames=frames)

    kept = out.details["tubes"]["kept"]
    assert len(kept) >= 1
    online = max(t["logit"] for t in kept)

    assert online == pytest.approx(offline, abs=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_model_parity.py -v`
Expected: FAIL — the `stabilize=True` case fails parity (online still crops per-frame because `predict` ignores the flag).

- [ ] **Step 3: Implement model.py change**

In `core/src/temporal_model/core/model.py`, in the `crop_tube_patches(...)` call inside `predict` (the keyword block ending with `normalization_std=mi["normalization"]["std"],`), add one line:

```python
                normalization_std=mi["normalization"]["std"],
                stabilize=mi.get("stabilize", True),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_model_parity.py -v`
Expected: PASS (both `stabilize=False` and `stabilize=True` parametrizations)

- [ ] **Step 5: Full core suite + lint, then commit**

Run: `cd core && uv run pytest -q && make lint`
Expected: all pass, ruff clean.

```bash
git add core/src/temporal_model/core/model.py core/tests/test_model_parity.py
git commit -m "feat(core): predict crops stabilized from baked config (default true)"
```

---

## Task 5: `build_model_input.py` — `--stabilize` CLI arg

**Files:**
- Modify: `train/src/temporal_model/train/build_model_input.py`
- Test: `train/tests/test_build_model_input.py` (create)

The DVC `build_model_input` stage passes crop params as CLI args. Add `--stabilize` (default `"true"`), parsed by a `_to_bool` helper, threaded into `process_tube`.

- [ ] **Step 1: Write the failing test**

Create `train/tests/test_build_model_input.py`:

```python
"""Tests for build_model_input CLI helpers + stabilize threading."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.train.build_model_input import _process_one, _to_bool


def test_to_bool_parses_dvc_strings():
    assert _to_bool("true") is True
    assert _to_bool("True") is True
    assert _to_bool("1") is True
    assert _to_bool("false") is False
    assert _to_bool("no") is False


def _write_jpg(path: Path, color: tuple[int, int, int], w: int = 320, h: int = 240):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), color, dtype=np.uint8)).save(path, format="JPEG")


def _write_tube(path: Path, seq_id: str, frame_ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sequence_id": seq_id,
                "split": "train",
                "label": "smoke",
                "source": "gt",
                "num_frames": len(frame_ids),
                "tube": {
                    "start_frame": 0,
                    "end_frame": len(frame_ids) - 1,
                    "entries": [
                        {
                            "frame_idx": i,
                            "frame_id": fid,
                            "bbox": [0.3 + 0.2 * i, 0.5, 0.1, 0.1],
                            "is_gap": False,
                            "confidence": 0.9,
                        }
                        for i, fid in enumerate(frame_ids)
                    ],
                },
            }
        )
    )


def test_process_one_threads_stabilize(tmp_path):
    seq_id = "site_1_2023-08-01T00-00-00"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(tmp_path / "raw" / "wildfire" / seq_id / "images" / f"{fid}.jpg",
                   (200, 30, 30))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    _write_tube(tube_path, seq_id, frame_ids)
    out_dir = tmp_path / "out"

    sid, label, err = _process_one(
        tube_path, tmp_path / "raw", out_dir,
        context_factor=1.5, patch_size=224, stabilize=True,
    )
    assert err is None and sid == seq_id and label == "smoke"
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd train && uv run pytest tests/test_build_model_input.py -v`
Expected: FAIL — `ImportError: cannot import name '_to_bool'` (and `_process_one` lacks `stabilize`).

- [ ] **Step 3: Implement**

In `train/src/temporal_model/train/build_model_input.py`:

Add the helper above `_process_one`:

```python
def _to_bool(value: str) -> bool:
    """Parse a DVC-substituted boolean param (``true``/``false``)."""
    return str(value).strip().lower() in {"true", "1", "yes"}
```

Add `stabilize: bool,` as the last param of `_process_one` and pass it into `process_tube`:

```python
def _process_one(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
    stabilize: bool,
) -> tuple[str | None, str, str | None]:
    """Worker: returns (sequence_id, label, error_or_none)."""
    try:
        record = json.loads(tube_path.read_text())
        sequence_id = record["sequence_id"]
        label = record["label"]
        process_tube(
            tube_path=tube_path,
            raw_dir=raw_dir,
            out_dir=out_dir,
            context_factor=context_factor,
            patch_size=patch_size,
            stabilize=stabilize,
        )
        return sequence_id, label, None
    except Exception as exc:
        return None, "", f"{tube_path.name}: {exc}"
```

In `main()`, add the arg after `--patch-size`:

```python
    parser.add_argument("--stabilize", default="true")
```

And pass it into each `pool.submit(...)` call as the final positional arg (after `args.patch_size`):

```python
        futures = [
            pool.submit(
                _process_one,
                p,
                args.raw_dir,
                args.output_dir,
                args.context_factor,
                args.patch_size,
                _to_bool(args.stabilize),
            )
            for p in tube_paths
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd train && uv run pytest tests/test_build_model_input.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/src/temporal_model/train/build_model_input.py train/tests/test_build_model_input.py
git commit -m "feat(train): --stabilize flag in build_model_input (default true)"
```

---

## Task 6: `package.py` — bake `stabilize` into the model config

**Files:**
- Modify: `train/src/temporal_model/train/package.py`
- Test: `train/tests/test_package_builders.py`

`build_config` builds the `model_input` block baked into `model.zip`. Add `stabilize`, read from the `model_input.stabilize` param, defaulting to `True` when absent.

- [ ] **Step 1: Write the failing test**

Append to `train/tests/test_package_builders.py`:

```python
def test_build_config_bakes_stabilize_from_param() -> None:
    params = {**PARAMS, "model_input": {**PARAMS["model_input"], "stabilize": True}}
    cfg = build_config(
        params,
        params["train_vit_dinov2_finetune"],
        threshold=0.4,
        aggregation="logistic",
        logistic_threshold=0.52,
    )
    assert cfg["model_input"]["stabilize"] is True


def test_build_config_stabilize_defaults_true_when_absent() -> None:
    # PARAMS["model_input"] has no "stabilize" key.
    cfg = build_config(
        PARAMS,
        PARAMS["train_vit_dinov2_finetune"],
        threshold=0.4,
        aggregation="logistic",
        logistic_threshold=0.52,
    )
    assert cfg["model_input"]["stabilize"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd train && uv run pytest tests/test_package_builders.py -k stabilize -v`
Expected: FAIL — `KeyError: 'stabilize'`

- [ ] **Step 3: Implement**

In `train/src/temporal_model/train/package.py`, in `build_config`'s returned `model_input` dict (currently `context_factor`, `patch_size`, `normalization`), add the `stabilize` key:

```python
        "model_input": {
            "context_factor": all_params["model_input"]["context_factor"],
            "patch_size": all_params["model_input"]["patch_size"],
            "stabilize": all_params["model_input"].get("stabilize", True),
            "normalization": _NORMALIZATION,
        },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd train && uv run pytest tests/test_package_builders.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add train/src/temporal_model/train/package.py train/tests/test_package_builders.py
git commit -m "feat(train): bake model_input.stabilize into packaged config"
```

---

## Task 7: `params.yaml` + `dvc.yaml` wiring (train)

**Files:**
- Modify: `train/params.yaml`
- Modify: `train/dvc.yaml`

- [ ] **Step 1: Add the param**

In `train/params.yaml`, under the `model_input:` block, add `stabilize: true`:

```yaml
model_input:
  context_factor: 1.5
  patch_size: 224
  stabilize: true
```

- [ ] **Step 2: Wire `build_model_input` to pass it + depend on stabilize.py**

In `train/dvc.yaml`, in the `build_model_input` stage's `do.cmd`, add the flag after `--patch-size`:

```yaml
        --context-factor ${model_input.context_factor}
        --patch-size ${model_input.patch_size}
        --stabilize ${model_input.stabilize}
```

In that stage's `deps`, add the stabilize module (after `../core/src/temporal_model/core/model_input.py`):

```yaml
        - ../core/src/temporal_model/core/stabilize.py
```

(The `params:` list already includes the whole `model_input` block, so `stabilize` is tracked automatically.)

- [ ] **Step 3: Add stabilize.py dep to the `package` stage**

In `train/dvc.yaml`, in the `package` stage's `deps`, add (after `../core/src/temporal_model/core/inference.py`):

```yaml
        - ../core/src/temporal_model/core/stabilize.py
```

- [ ] **Step 4: Verify DVC parses the pipeline**

Run: `cd train && uv run dvc status build_model_input 2>&1 | head` (or `uv run dvc dag` if `dvc status` requires data)
Expected: DVC parses the YAML without error (it will likely report the stage as "changed"/needs-repro — that is correct, not a failure). If `dvc` is unavailable, instead run:
`cd train && python -c "import yaml; yaml.safe_load(open('dvc.yaml')); yaml.safe_load(open('params.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add train/params.yaml train/dvc.yaml
git commit -m "feat(train): default model_input.stabilize=true and wire dvc stages"
```

---

## Task 8: `eval/dvc.yaml` — track stabilize.py dependency

**Files:**
- Modify: `eval/dvc.yaml`

Eval has no behavioural change (it crops from the baked `model.zip` config), but `inference.py` now imports `stabilize.py`, so list it as a dep for hash correctness.

- [ ] **Step 1: Add the dep**

In `eval/dvc.yaml`, in the `evaluate` stage's `deps`, add (after `../core/src/temporal_model/core/inference.py`):

```yaml
        - ../core/src/temporal_model/core/stabilize.py
```

- [ ] **Step 2: Verify DVC parses**

Run: `cd eval && python -c "import yaml; yaml.safe_load(open('dvc.yaml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add eval/dvc.yaml
git commit -m "chore(eval): track core/stabilize.py as evaluate dependency"
```

---

## Task 9: Full verification across packages

**Files:** none (verification only)

- [ ] **Step 1: core — tests + lint**

Run: `cd core && uv run pytest -q && make lint`
Expected: all pass, ruff clean.

- [ ] **Step 2: train — tests + lint**

Run: `cd train && uv run pytest -q && make lint`
Expected: all pass, ruff clean.

- [ ] **Step 3: eval — tests + lint**

Run: `cd eval && uv run pytest -q && make lint`
Expected: all pass, ruff clean.

- [ ] **Step 4: Confirm no stray references**

Run: `git grep -n "stabilize" core/src train/src eval/src train/params.yaml train/dvc.yaml eval/dvc.yaml`
Expected: `stabilize` appears in `stabilize.py`, `model_input.py`, `inference.py`, `model.py`, `build_model_input.py`, `package.py`, `params.yaml`, `train/dvc.yaml`, `eval/dvc.yaml` — and nowhere it shouldn't.

---

## Done criteria

- New `core/stabilize.py` with `union_window`, fully unit-tested.
- `process_tube`, `crop_tube_patches`, and `predict` crop stabilized by default; `stabilize=False` reproduces the old per-frame behaviour (pinned by tests).
- `model.zip` bakes `stabilize: true`; eval reports stabilized metrics with no eval code change.
- `make test` + `make lint` green in `core`, `train`, `eval`.
- The downstream `dvc repro` (regenerating crops, checkpoint, `model.zip`, and eval reports) is a **separate post-merge step**, not part of this plan.
