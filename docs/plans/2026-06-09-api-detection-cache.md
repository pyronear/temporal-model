# API Detection Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the API from re-running YOLO on frames a growing sequence has already seen, by caching per-frame detections (keyed by stable `frame_id`) in the API and reusing them on later calls.

**Architecture:** `core` gains a pure detection-injection seam — a public `detect()` and a `frame_detections=` argument to `predict()` — with no state. The API (`ModelRunner`) owns a size-bounded LRU keyed by `frame_id`, detects only cache-miss frames per request, and passes the resolved detections into `predict()`. `core` stays deterministic so `benchmark`/`eval` are unaffected.

**Tech Stack:** Python 3.13, FastAPI, PyTorch/ultralytics (mocked in tests), pytest, `uv`. Per-package tests: `cd <pkg> && uv run pytest`.

**Spec:** `docs/specs/2026-06-09-api-detection-cache-design.md`

---

## File Structure

- `core/src/temporal_model/core/model.py` — add `detect()`, `_resolve_frame_detections()`, and the `frame_detections=` branch in `predict()`. (Modify)
- `core/tests/test_detection_injection.py` — core seam tests. (Create)
- `api/src/temporal_model/api/detection_cache.py` — the LRU. (Create)
- `api/tests/test_detection_cache.py` — LRU tests. (Create)
- `api/src/temporal_model/api/settings.py` — add `detection_cache_size`. (Modify)
- `api/tests/test_settings.py` — settings test. (Modify)
- `api/src/temporal_model/api/model_runner.py` — wire the cache into `predict`. (Modify)
- `api/tests/test_model_runner.py` — runner orchestration tests. (Modify)
- `api/src/temporal_model/api/app.py` — pass the setting into `ModelRunner.load`. (Modify)

---

## Task 1: `core` detection-injection seam

**Files:**
- Modify: `core/src/temporal_model/core/model.py`
- Test: `core/tests/test_detection_injection.py`

### Background for the implementer

`predict(self, frames, *, timer=None, compute_trigger=False)` (`core/model.py:145`) runs a `detector` stage at ~`core/model.py:223`:

```python
        with stage_ctx(timer, "detector"):
            frame_dets = run_yolo_on_frames(
                self._yolo,
                truncated,
                confidence_threshold=infer["confidence_threshold"],
                iou_nms=infer["iou_nms"],
                image_size=infer["image_size"],
                device=self._device,
            )
```

`run_yolo_on_frames(yolo, frames, …) -> list[FrameDetections]` (`core/inference.py:114`) returns one `FrameDetections` per frame with `frame_idx` = position. `FrameDetections` (`core/types.py`) is a dataclass `{frame_idx, frame_id, timestamp, detections}`. `Frame.frame_id` is the filename stem.

We add a public `detect()` (the existing detector call, extracted) and let `predict()` accept a `frame_detections` mapping: reuse provided entries, run YOLO only on the misses, and **re-stamp** each entry's positional `frame_idx`.

- [ ] **Step 1: Write the failing tests**

Create `core/tests/test_detection_injection.py`:

```python
"""Tests for the pure detection-injection seam on BboxTubeTemporalModel."""

from unittest.mock import MagicMock

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.types import FrameDetections

# Reuse the shared fixtures/helpers from the edge-case suite.
from test_model_edge_cases import (  # type: ignore[import-not-found]
    TEST_CONFIG,
    _fake_yolo_factory,
    red_frames,
    tiny_classifier,
)

__all__ = ["red_frames", "tiny_classifier"]  # keep fixtures importable


def test_detect_returns_one_framedetections_per_frame(red_frames, tiny_classifier):
    per_frame = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    dets = model.detect(red_frames)

    assert [d.frame_id for d in dets] == [f.frame_id for f in red_frames]
    assert [d.frame_idx for d in dets] == list(range(len(red_frames)))
    assert all(len(d.detections) == 1 for d in dets)


def test_injection_parity_matches_full_detection(red_frames, tiny_classifier):
    per_frame = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    baseline = model.predict(frames=red_frames)
    cached = {fd.frame_id: fd for fd in model.detect(red_frames)}
    injected = model.predict(frames=red_frames, frame_detections=cached)

    assert injected == baseline


def test_injection_detects_only_misses(red_frames, tiny_classifier):
    # All but the last frame are pre-supplied; YOLO must run on the last only.
    cached = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
        )
        for i, f in enumerate(red_frames[:-1])
    }
    yolo = _fake_yolo_factory([[]])  # asserts it is called with exactly 1 frame
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    model.predict(frames=red_frames, frame_detections=cached)

    yolo.predict.assert_called_once()
    called_paths = yolo.predict.call_args[0][0]
    assert called_paths == [str(red_frames[-1].image_path)]


def test_injection_all_cached_skips_yolo(red_frames, tiny_classifier):
    cached = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
        )
        for i, f in enumerate(red_frames)
    }
    yolo = MagicMock()
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    model.predict(frames=red_frames, frame_detections=cached)

    yolo.predict.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_detection_injection.py -v`
Expected: FAIL — `AttributeError: 'BboxTubeTemporalModel' object has no attribute 'detect'` and `predict() got an unexpected keyword argument 'frame_detections'`.

- [ ] **Step 3: Add imports to `model.py`**

At the top of `core/src/temporal_model/core/model.py`, add the dataclass `replace` and the `FrameDetections` type. Add this import line (near the other `from .` imports):

```python
from dataclasses import replace
```

and add `FrameDetections` via a new import line:

```python
from .types import FrameDetections
```

- [ ] **Step 4: Add `detect()` and `_resolve_frame_detections()`**

Insert these two methods into `BboxTubeTemporalModel`, immediately **before** `def predict(` (around `core/model.py:145`):

```python
    def detect(self, frames: list[Frame]) -> list[FrameDetections]:
        """Run the companion YOLO detector over ``frames`` (one batched call).

        Pure: same input → same output. Exposed so a serving layer can cache
        per-frame detections and avoid re-detecting frames it has already seen.
        """
        infer = self._cfg["infer"]
        return run_yolo_on_frames(
            self._yolo,
            frames,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )

    def _resolve_frame_detections(
        self,
        truncated: list[Frame],
        frame_detections: dict[str, FrameDetections],
    ) -> list[FrameDetections]:
        """Use supplied detections where present, detect the rest, in order.

        Each entry's positional ``frame_idx`` is re-stamped to its index in
        ``truncated`` — a cached entry carries the ``frame_idx`` from the call
        that produced it, which is meaningless here.
        """
        misses = [f for f in truncated if f.frame_id not in frame_detections]
        fresh = {fd.frame_id: fd for fd in self.detect(misses)}
        resolved: list[FrameDetections] = []
        for idx, f in enumerate(truncated):
            fd = frame_detections.get(f.frame_id)
            if fd is None:
                fd = fresh[f.frame_id]
            resolved.append(replace(fd, frame_idx=idx))
        return resolved
```

- [ ] **Step 5: Add the `frame_detections` parameter and branch in `predict()`**

Change the `predict` signature (`core/model.py:145`) from:

```python
    def predict(
        self,
        frames: list[Frame],
        *,
        timer: StageTimer | None = None,
        compute_trigger: bool = False,
    ) -> TemporalModelOutput:
```

to:

```python
    def predict(
        self,
        frames: list[Frame],
        *,
        frame_detections: dict[str, FrameDetections] | None = None,
        timer: StageTimer | None = None,
        compute_trigger: bool = False,
    ) -> TemporalModelOutput:
```

Then replace the detector stage block (`core/model.py:223`) from:

```python
        with stage_ctx(timer, "detector"):
            frame_dets = run_yolo_on_frames(
                self._yolo,
                truncated,
                confidence_threshold=infer["confidence_threshold"],
                iou_nms=infer["iou_nms"],
                image_size=infer["image_size"],
                device=self._device,
            )
```

to:

```python
        with stage_ctx(timer, "detector"):
            if frame_detections is None:
                frame_dets = self.detect(truncated)
            else:
                frame_dets = self._resolve_frame_detections(
                    truncated, frame_detections
                )
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `cd core && uv run pytest tests/test_detection_injection.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full core suite (parity guardrail)**

Run: `cd core && uv run pytest tests/ -q`
Expected: PASS — existing `test_model_parity`, `test_model_edge_cases`, etc. unchanged.

- [ ] **Step 8: Lint**

Run: `cd core && uv run ruff check .`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add core/src/temporal_model/core/model.py core/tests/test_detection_injection.py
git commit -m "feat(core): pure detection-injection seam (detect + frame_detections)"
```

---

## Task 2: API setting `detection_cache_size`

**Files:**
- Modify: `api/src/temporal_model/api/settings.py`
- Test: `api/tests/test_settings.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_settings.py`:

```python
def test_detection_cache_size_default():
    from temporal_model.api.settings import Settings

    assert Settings().detection_cache_size == 4096


def test_detection_cache_size_env_override(monkeypatch):
    from temporal_model.api.settings import Settings

    monkeypatch.setenv("TEMPORAL_API_DETECTION_CACHE_SIZE", "10")
    assert Settings().detection_cache_size == 10
```

- [ ] **Step 2: Run to verify failure**

Run: `cd api && uv run pytest tests/test_settings.py -k detection_cache_size -v`
Expected: FAIL — `AttributeError`/no such field `detection_cache_size`.

- [ ] **Step 3: Add the field**

In `api/src/temporal_model/api/settings.py`, add this field to `Settings` (after `calibrator_threshold`):

```python
    # Per-frame detection LRU capacity (frame_id → detections). 0 disables.
    detection_cache_size: int = 4096
```

- [ ] **Step 4: Run to verify pass**

Run: `cd api && uv run pytest tests/test_settings.py -k detection_cache_size -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/settings.py api/tests/test_settings.py
git commit -m "feat(api): add TEMPORAL_API_DETECTION_CACHE_SIZE setting"
```

---

## Task 3: API `DetectionCache` (size-bounded LRU)

**Files:**
- Create: `api/src/temporal_model/api/detection_cache.py`
- Test: `api/tests/test_detection_cache.py`

### Notes

A tiny LRU over `OrderedDict`. Values are stored as-is (no `core` type import needed — keeps the runner's import light). `capacity <= 0` disables it: `put` is a no-op so `__contains__` is always `False` and every frame becomes a miss (→ full detection, identical to pre-change behavior).

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_detection_cache.py`:

```python
from temporal_model.api.detection_cache import DetectionCache


def test_put_get_contains():
    c = DetectionCache(capacity=4)
    c.put("a", 1)
    assert "a" in c
    assert c.get("a") == 1
    assert len(c) == 1


def test_evicts_least_recently_used():
    c = DetectionCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts "a"
    assert "a" not in c
    assert "b" in c and "c" in c
    assert len(c) == 2


def test_get_marks_recently_used():
    c = DetectionCache(capacity=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")      # "a" now most-recently-used
    c.put("c", 3)   # evicts "b", not "a"
    assert "a" in c
    assert "b" not in c
    assert "c" in c


def test_capacity_zero_disables():
    c = DetectionCache(capacity=0)
    c.put("a", 1)
    assert "a" not in c
    assert len(c) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd api && uv run pytest tests/test_detection_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: temporal_model.api.detection_cache`.

- [ ] **Step 3: Implement the cache**

Create `api/src/temporal_model/api/detection_cache.py`:

```python
"""A small size-bounded LRU for per-frame detections.

Keyed by ``frame_id``. Values are stored opaquely (the caller decides the type).
``capacity <= 0`` disables the cache: nothing is stored, so every lookup misses.
"""

from collections import OrderedDict
from typing import Any


class DetectionCache:
    """Least-recently-used cache with a fixed maximum entry count."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Any:
        """Return the value for ``key`` and mark it most-recently-used."""
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Insert/update ``key``; evict the LRU entry if over capacity."""
        if self._capacity <= 0:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd api && uv run pytest tests/test_detection_cache.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `cd api && uv run ruff check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add api/src/temporal_model/api/detection_cache.py api/tests/test_detection_cache.py
git commit -m "feat(api): size-bounded detection LRU cache"
```

---

## Task 4: Wire the cache into `ModelRunner`

**Files:**
- Modify: `api/src/temporal_model/api/model_runner.py`
- Test: `api/tests/test_model_runner.py`

### Background for the implementer

Today `ModelRunner.predict` (`api/model_runner.py:111`) calls `self._model.predict_sequence(frame_paths)` in a threadpool under a lock. We replace that with an orchestration that: loads frames, splits cache hits/misses, detects misses, updates the cache, and calls `predict(frames, frame_detections=…)`. All of it runs in the worker thread under the existing lock, so cache access is serialized and thread-safe.

`load`'s signature gains `detection_cache_size`; `__init__` builds the cache.

- [ ] **Step 1: Update the existing delegation test and add cache tests**

In `api/tests/test_model_runner.py`, **replace** `test_predict_delegates_to_model` (lines ~113-125) with the following, and add the two new tests. These use a fake model exposing the new `load_sequence`/`detect`/`predict` surface.

```python
from pathlib import Path
from types import SimpleNamespace

from temporal_model.core.protocol import Frame
from temporal_model.core.types import FrameDetections


class _OrchestrationModel:
    """Fake core model recording how detection is invoked across calls."""

    def __init__(self):
        self.detect_calls: list[list[str]] = []
        self.predict_calls: list[set[str]] = []

    def load_sequence(self, paths):
        return [
            Frame(frame_id=Path(p).stem, image_path=Path(p), timestamp=None)
            for p in paths
        ]

    def detect(self, frames):
        self.detect_calls.append([f.frame_id for f in frames])
        return [
            FrameDetections(
                frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
            )
            for i, f in enumerate(frames)
        ]

    def predict(self, frames, *, frame_detections=None):
        self.predict_calls.append(set(frame_detections or {}))
        return SimpleNamespace(frame_ids=[f.frame_id for f in frames])


def test_predict_resolves_all_detections_for_model():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    out = asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))

    assert out.frame_ids == ["x_00", "x_01"]
    # predict() receives detections for every frame in the sequence.
    assert model.predict_calls[-1] == {"x_00", "x_01"}


def test_predict_caches_and_reuses_detections():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=4096
    )
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg", "c/x_02.jpg"]))

    assert model.detect_calls[0] == ["x_00", "x_01"]
    assert model.detect_calls[1] == ["x_02"]  # only the new frame re-detected


def test_predict_cache_disabled_detects_every_frame():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=0
    )
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg", "c/x_02.jpg"]))

    assert model.detect_calls[0] == ["x_00", "x_01"]
    assert model.detect_calls[1] == ["x_00", "x_01", "x_02"]  # full each call
```

- [ ] **Step 2: Run to verify failure**

Run: `cd api && uv run pytest tests/test_model_runner.py -v`
Expected: FAIL — `ModelRunner.__init__` has no `detection_cache_size`; the fake model has no `predict_sequence`.

- [ ] **Step 3: Import the cache and add the constructor arg**

In `api/src/temporal_model/api/model_runner.py`, add `import time` to the stdlib imports (next to `import asyncio`), and add the cache import near the top (after the existing imports):

```python
from .detection_cache import DetectionCache
```

Change `__init__` (`api/model_runner.py:46`) to accept `detection_cache_size` and build the cache. Replace the signature/body:

```python
    def __init__(
        self,
        model: Any,
        *,
        name: str,
        version: str | None,
        calibrated: bool,
        threshold_overridden: bool = False,
        packaged_threshold: float | None = None,
        detection_cache_size: int = 0,
    ) -> None:
        self._model = model
        self.name = name
        self.version = version
        self.calibrated = calibrated
        self.threshold_overridden = threshold_overridden
        self.packaged_threshold = packaged_threshold
        self._cache = DetectionCache(detection_cache_size)
        self._lock = asyncio.Lock()
```

- [ ] **Step 4: Thread the setting through `load()`**

Change `load`'s signature (`api/model_runner.py:64`) to add the parameter:

```python
    @classmethod
    def load(
        cls,
        package_path: Path,
        device: str | None,
        calibrator_threshold: float | None = None,
        detection_cache_size: int = 0,
    ) -> "ModelRunner":
```

and pass it into the `return cls(...)` at the end of `load` (`api/model_runner.py:104`):

```python
        return cls(
            model,
            **meta,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
            detection_cache_size=detection_cache_size,
        )
```

- [ ] **Step 5: Replace `predict()` with the cache orchestration**

Replace `predict` (`api/model_runner.py:111-114`) with:

```python
    async def predict(self, frame_paths: list[Path]) -> Any:
        """Resolve detections (cache + detect misses) then run the model.

        The whole orchestration runs in a worker thread under the lock, so the
        cache is accessed by one prediction at a time.
        """
        async with self._lock:
            return await run_in_threadpool(self._predict_sync, frame_paths)

    def _predict_sync(self, frame_paths: list[Path]) -> Any:
        started = time.perf_counter()
        frames = self._model.load_sequence(frame_paths)
        resolved: dict[str, Any] = {}
        misses = []
        for f in frames:
            if f.frame_id in self._cache:
                resolved[f.frame_id] = self._cache.get(f.frame_id)
            else:
                misses.append(f)
        for fd in self._model.detect(misses):
            self._cache.put(fd.frame_id, fd)
            resolved[fd.frame_id] = fd
        out = self._model.predict(frames, frame_detections=resolved)
        logger.info(
            "predict: %d/%d cache hits, seq_len=%d, cache_size=%d, %.0fms",
            len(frames) - len(misses),
            len(frames),
            len(frames),
            len(self._cache),
            (time.perf_counter() - started) * 1000.0,
        )
        return out
```

- [ ] **Step 6: Run the runner tests to verify they pass**

Run: `cd api && uv run pytest tests/test_model_runner.py -v`
Expected: PASS (all, including the rewritten delegation test).

- [ ] **Step 7: Run the full API suite**

Run: `cd api && uv run pytest tests/ -q`
Expected: PASS — `test_app.py` integration (model mocked, S3 via moto) still green.

- [ ] **Step 8: Lint**

Run: `cd api && uv run ruff check .`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add api/src/temporal_model/api/model_runner.py api/tests/test_model_runner.py
git commit -m "feat(api): cache per-frame detections in ModelRunner, detect only misses"
```

---

## Task 5: Pass the setting from app startup

**Files:**
- Modify: `api/src/temporal_model/api/app.py`

### Background

`lifespan` (`api/app.py:50`) builds the runner via `ModelRunner.load(...)`. Add the new setting so production actually enables the cache.

- [ ] **Step 1: Update the `ModelRunner.load` call**

In `api/src/temporal_model/api/app.py`, change the `lifespan` load call from:

```python
        app.state.runner = ModelRunner.load(
            Path(settings.model_path),
            settings.device,
            settings.calibrator_threshold,
        )
```

to:

```python
        app.state.runner = ModelRunner.load(
            Path(settings.model_path),
            settings.device,
            settings.calibrator_threshold,
            detection_cache_size=settings.detection_cache_size,
        )
```

- [ ] **Step 2: Run the full API suite**

Run: `cd api && uv run pytest tests/ -q`
Expected: PASS — the app boots with the cache enabled; existing tests unchanged.

- [ ] **Step 3: Lint**

Run: `cd api && uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add api/src/temporal_model/api/app.py
git commit -m "feat(api): enable detection cache at startup from settings"
```

---

## Final verification

- [ ] **Run core + api suites**

Run: `make -C core test && make -C api test`
Expected: all PASS.

- [ ] **Confirm `core` default path is untouched (parity guardrail)**

Run: `cd core && uv run pytest tests/test_model_parity.py -q`
Expected: PASS — proves `predict(frames)` (no injection) is unchanged, so `benchmark`/`eval` are unaffected.

---

## Spec coverage check

- Core pure seam (`detect()` + `frame_detections=`, re-stamped `frame_idx`, miss-only YOLO) → Task 1.
- `frame_id`-keyed size-bounded LRU, default 4096, `0` disables → Tasks 2 + 3.
- Cache lives on `ModelRunner`, reset on reload (new instance), accessed under the lock → Task 4.
- Cache-hit + latency observability → Task 4 (the `logger.info` line; sequence length and hit count logged).
- `predict_sequence`/protocol not threaded with the flag; default path bit-for-bit identical → Task 1 (parity test) + final verification.
- Startup wiring so prod enables it → Task 5.
