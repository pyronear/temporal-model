# Optional ROI (`roi_xyxyn`) on `/predict` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional normalized region-of-interest to the temporal model API so the verdict is computed only over tubes intersecting the ROI.

**Architecture:** Tubes are built from all detections as today; right after `build_tubes_for_inference()` (before crop/classify/calibrate), tubes with no real detection intersecting the ROI are dropped. The filter lives in `core` (`BboxTubeTemporalModel.predict(roi=...)`); the API adds a `roi_xyxyn: [x_min, y_min, x_max, y_max]` request field and threads a plain tuple down through `ModelRunner`. The detection LRU cache is untouched — it stores full-frame detections upstream of the filter (this invariant must hold; see spec).

**Spec:** `docs/specs/2026-06-10-api-roi-design.md` — read it before starting.

**Tech Stack:** Python 3.11+, FastAPI/Pydantic, pytest, `uv` (run tests via `make -C core test` / `make -C api test`, or `uv run pytest tests/...` from the package dir).

**Conventions:** Conventional-commit messages. Do NOT add Claude co-author trailers to commits.

---

## File Structure

| File | Change |
|---|---|
| `core/src/temporal_model/core/tubes.py` | Add `tube_intersects_roi()` helper + `__all__` entry |
| `core/src/temporal_model/core/details_schema.py` | `Tubes` gains `num_outside_roi: int = 0` |
| `core/src/temporal_model/core/model.py` | `predict()` gains `roi` param: validate, filter kept tubes, report count |
| `core/tests/test_roi.py` | New: helper unit tests + `predict(roi=...)` behavior tests |
| `core/tests/test_details_schema.py` | Back-compat tests for the new schema field |
| `api/src/temporal_model/api/schemas.py` | `PredictRequest.roi_xyxyn` + validator; `Preprocessing.num_tubes_outside_roi` + mapping |
| `api/src/temporal_model/api/model_runner.py` | Thread `roi` through `predict`/`_predict_sync` |
| `api/src/temporal_model/api/app.py` | Pass `roi=body.roi_xyxyn` to the runner |
| `api/tests/test_schemas.py` | Request validation + response mapping tests |
| `api/tests/test_model_runner.py` | Update fakes' `predict` signatures; roi threading tests |
| `api/tests/test_app.py` | Update `FakeRunner`; HTTP-level roi tests |
| `api/README.md` | Document the new body field |

---

### Task 1: `tube_intersects_roi` helper in core

**Files:**
- Create: `core/tests/test_roi.py`
- Modify: `core/src/temporal_model/core/tubes.py`

- [ ] **Step 1: Write the failing tests**

Create `core/tests/test_roi.py`:

```python
"""Tests for ROI tube filtering (spec: 2026-06-10-api-roi-design.md)."""

from temporal_model.core.tubes import tube_intersects_roi
from temporal_model.core.types import Detection, Tube, TubeEntry


def _det(cx: float, cy: float, w: float = 0.1, h: float = 0.1, conf: float = 0.8):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


def _tube(entries: list[TubeEntry]) -> Tube:
    return Tube(tube_id=0, entries=entries, start_frame=0, end_frame=len(entries))


class TestTubeIntersectsRoi:
    def test_overlapping_detection_keeps_tube(self):
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.55, 0.45, 0.9, 0.55)) is True

    def test_touching_edge_counts_as_overlap(self):
        # Detection box right edge at x=0.6 exactly touches roi x_min=0.6.
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.6, 0.4, 0.9, 0.6)) is True

    def test_fully_outside_drops_tube(self):
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.7, 0.7, 0.9, 0.9)) is False

    def test_gap_entries_do_not_count(self):
        # The only entry overlapping the ROI is a gap (synthetic, lerped bbox);
        # the sole real detection is outside. Tube must be dropped.
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=_det(0.5, 0.5), is_gap=True),
                TubeEntry(frame_idx=1, detection=_det(0.1, 0.1)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is False

    def test_pre_interpolation_gap_without_detection_is_ignored(self):
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=None, is_gap=True),
                TubeEntry(frame_idx=1, detection=_det(0.5, 0.5)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is True

    def test_any_single_real_entry_inside_keeps_tube(self):
        # First entries outside, last one drifted into the ROI.
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=_det(0.1, 0.1)),
                TubeEntry(frame_idx=1, detection=_det(0.2, 0.2)),
                TubeEntry(frame_idx=2, detection=_det(0.5, 0.5)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_roi.py -v`
Expected: FAIL — `ImportError: cannot import name 'tube_intersects_roi'`

- [ ] **Step 3: Implement the helper**

In `core/src/temporal_model/core/tubes.py`, add `"tube_intersects_roi"` to `__all__` and append the function at the end of the file:

```python
def tube_intersects_roi(
    tube: Tube, roi: tuple[float, float, float, float]
) -> bool:
    """True if any real (non-gap) detection bbox overlaps the ROI rectangle.

    ``roi`` is ``(x_min, y_min, x_max, y_max)`` normalized to [0, 1];
    touching edges count as overlap. Gap entries are synthetic (interpolated)
    and do not count.
    """
    x_min, y_min, x_max, y_max = roi
    for entry in tube.entries:
        if entry.is_gap or entry.detection is None:
            continue
        d = entry.detection
        if (
            d.cx - d.w / 2 <= x_max
            and d.cx + d.w / 2 >= x_min
            and d.cy - d.h / 2 <= y_max
            and d.cy + d.h / 2 >= y_min
        ):
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd core && uv run pytest tests/test_roi.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/tubes.py core/tests/test_roi.py
git commit -m "feat(core): tube_intersects_roi helper for ROI tube filtering"
```

---

### Task 2: `num_outside_roi` on the details schema

**Files:**
- Modify: `core/src/temporal_model/core/details_schema.py:52-54`
- Modify: `core/tests/test_details_schema.py`

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_details_schema.py` (it already imports `BboxTubeDetails`, `Decision`, `Preprocessing`, `Tubes`):

```python
def test_tubes_num_outside_roi_defaults_to_zero():
    assert Tubes(num_candidates=2, kept=[]).num_outside_roi == 0


def test_details_parses_legacy_dump_without_num_outside_roi():
    # Dumps serialized before the ROI feature must still validate.
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=1, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=0, kept=[]),
        decision=Decision(
            aggregation="max_logit", threshold=0.0, trigger_tube_id=None
        ),
    )
    dump = details.model_dump()
    del dump["tubes"]["num_outside_roi"]
    parsed = BboxTubeDetails.model_validate(dump)
    assert parsed.tubes.num_outside_roi == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_details_schema.py -v -k num_outside_roi`
Expected: FAIL — `Tubes` has no attribute/field `num_outside_roi` (the legacy test fails on the `del` of a missing key)

- [ ] **Step 3: Add the field**

In `core/src/temporal_model/core/details_schema.py`:

```python
class Tubes(_Frozen):
    num_candidates: int
    num_outside_roi: int = 0
    kept: list[KeptTube]
```

- [ ] **Step 4: Run the full schema test file**

Run: `cd core && uv run pytest tests/test_details_schema.py -v`
Expected: all PASS (pre-existing tests construct `Tubes` with kwargs, so the new default is non-breaking)

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/details_schema.py core/tests/test_details_schema.py
git commit -m "feat(core): num_outside_roi field on details tubes block"
```

---

### Task 3: `roi` parameter on `BboxTubeTemporalModel.predict()`

**Files:**
- Modify: `core/src/temporal_model/core/model.py` (imports ~line 39, `predict()` lines 189-309)
- Modify: `core/tests/test_roi.py`

- [ ] **Step 1: Write the failing tests**

Append to `core/tests/test_roi.py`. Fixture/helper reuse follows the existing pattern in `core/tests/test_detection_injection.py` (pytest collects sibling test modules on `sys.path`):

```python
from unittest.mock import MagicMock

import pytest

# Reuse the shared fixtures/helpers from the edge-case suite.
from test_model_edge_cases import (  # type: ignore[import-not-found]
    TEST_CONFIG,
    _fake_yolo_factory,
    red_frames,
    tiny_classifier,
)

from temporal_model.core.model import BboxTubeTemporalModel

__all__ = ["red_frames", "tiny_classifier"]  # keep fixtures importable
```

(Put these imports at the top of the file with the existing ones; module-level `__all__` once.)

```python
def _two_cluster_model(red_frames, tiny_classifier):
    """Model whose fake YOLO emits two spatial clusters -> two tubes."""
    per_frame = [
        [(0.2, 0.5, 0.1, 0.1, 0.9), (0.7, 0.5, 0.1, 0.1, 0.9)]
        for _ in red_frames
    ]
    return BboxTubeTemporalModel(
        yolo_model=_fake_yolo_factory(per_frame),
        classifier=tiny_classifier,
        config=TEST_CONFIG,
    )


class TestPredictWithRoi:
    def test_roi_keeps_only_intersecting_tube(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        out = model.predict(frames=red_frames, roi=(0.6, 0.4, 0.8, 0.6))

        kept = out.details["tubes"]["kept"]
        assert len(kept) == 1
        assert out.details["tubes"]["num_outside_roi"] == 1
        # The surviving tube is the x~0.7 cluster.
        cxs = [e["bbox"][0] for e in kept[0]["entries"] if e["bbox"]]
        assert all(abs(cx - 0.7) < 0.05 for cx in cxs)
        # num_candidates keeps its pre-ROI meaning.
        assert out.details["tubes"]["num_candidates"] == 2

    def test_roi_excluding_everything_is_negative(
        self, red_frames, tiny_classifier
    ):
        model = _two_cluster_model(red_frames, tiny_classifier)
        out = model.predict(frames=red_frames, roi=(0.45, 0.05, 0.55, 0.15))

        assert out.is_positive is False
        assert out.details["tubes"]["kept"] == []
        assert out.details["tubes"]["num_outside_roi"] == 2

    def test_roi_none_matches_baseline(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        baseline = model.predict(frames=red_frames)
        out = model.predict(frames=red_frames, roi=None)

        assert out == baseline
        assert baseline.details["tubes"]["num_outside_roi"] == 0

    def test_whole_frame_roi_matches_baseline(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        baseline = model.predict(frames=red_frames)
        out = model.predict(frames=red_frames, roi=(0.0, 0.0, 1.0, 1.0))

        assert out == baseline


class TestRoiValidation:
    @pytest.mark.parametrize(
        "roi",
        [
            (-0.1, 0.0, 1.0, 1.0),  # out of range low
            (0.0, 0.0, 1.0, 1.1),  # out of range high
            (0.5, 0.2, 0.4, 0.8),  # x_min >= x_max
            (0.2, 0.8, 0.4, 0.8),  # y_min >= y_max (zero height)
        ],
    )
    def test_invalid_roi_raises(self, roi, tiny_classifier):
        model = BboxTubeTemporalModel(
            yolo_model=MagicMock(),
            classifier=tiny_classifier,
            config=TEST_CONFIG,
        )
        with pytest.raises(ValueError, match="roi"):
            model.predict(frames=[], roi=roi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_roi.py -v`
Expected: Task 1 tests still PASS; new tests FAIL — `predict() got an unexpected keyword argument 'roi'`

- [ ] **Step 3: Implement in `model.py`**

3a. Extend the import from `.tubes` (line 39):

```python
from .tubes import build_tubes, tube_intersects_roi
```

3b. Add the parameter to `predict()` (line 189):

```python
    def predict(
        self,
        frames: list[Frame],
        *,
        frame_detections: dict[str, FrameDetections] | None = None,
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        compute_trigger: bool = False,
    ) -> TemporalModelOutput:
```

3c. Validate at the very top of `predict()`, before the config reads (so even `frames=[]` rejects a bad ROI):

```python
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            if (
                not all(0.0 <= c <= 1.0 for c in roi)
                or x_min >= x_max
                or y_min >= y_max
            ):
                raise ValueError(
                    f"invalid roi {roi!r}: expected normalized "
                    "(x_min, y_min, x_max, y_max) with x_min < x_max "
                    "and y_min < y_max"
                )
```

3d. Give `_make_details` a `num_outside_roi` parameter and pass it to `Tubes`:

```python
        def _make_details(
            *,
            num_frames_input: int,
            num_truncated: int,
            padded_indices: list[int],
            num_candidates: int,
            num_outside_roi: int,
            kept_tubes_models: list[KeptTube],
            trigger_tube_id: int | None,
        ) -> dict:
            return BboxTubeDetails(
                preprocessing=Preprocessing(
                    num_frames_input=num_frames_input,
                    num_truncated=num_truncated,
                    padded_frame_indices=padded_indices,
                ),
                tubes=Tubes(
                    num_candidates=num_candidates,
                    num_outside_roi=num_outside_roi,
                    kept=kept_tubes_models,
                ),
                decision=Decision(
                    aggregation=aggregation,
                    threshold=effective_threshold,
                    trigger_tube_id=trigger_tube_id,
                ),
            ).model_dump()
```

There are three `_make_details(...)` call sites; update each:
- empty-frames early return (~line 241): add `num_outside_roi=0,`
- no-kept-tubes early return (~line 301): add `num_outside_roi=num_outside_roi,`
- final return (~line 415): add `num_outside_roi=num_outside_roi,`

3e. Filter right after `build_tubes_for_inference(...)` inside the `tubes` stage block (after line 295), so the no-kept early return reflects post-ROI state:

```python
            num_outside_roi = 0
            if roi is not None:
                n_before = len(kept)
                kept = [t for t in kept if tube_intersects_roi(t, roi)]
                num_outside_roi = n_before - len(kept)
```

- [ ] **Step 4: Run the core suite**

Run: `cd core && uv run pytest tests/ -v`
Expected: all PASS (existing callers are unaffected: `roi` is keyword-only with default `None`, and `num_outside_roi=0` flows through legacy paths)

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/model.py core/tests/test_roi.py
git commit -m "feat(core): optional roi on predict() filters tubes before scoring"
```

---

### Task 4: `roi_xyxyn` request field on the API

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py:20-46`
- Modify: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_schemas.py`:

```python
def test_request_roi_defaults_to_none():
    assert PredictRequest(frames=["a.jpg"]).roi_xyxyn is None


def test_request_accepts_valid_roi():
    req = PredictRequest(frames=["a.jpg"], roi_xyxyn=[0.1, 0.2, 0.3, 0.4])
    assert req.roi_xyxyn == (0.1, 0.2, 0.3, 0.4)


def test_request_accepts_whole_frame_roi():
    req = PredictRequest(frames=["a.jpg"], roi_xyxyn=[0.0, 0.0, 1.0, 1.0])
    assert req.roi_xyxyn == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "roi",
    [
        [-0.1, 0.2, 0.3, 0.4],  # out of range low
        [0.1, 0.2, 0.3, 1.4],  # out of range high
        [0.3, 0.2, 0.1, 0.4],  # x_min >= x_max
        [0.1, 0.4, 0.3, 0.4],  # y_min >= y_max (zero height)
        [0.1, 0.2, 0.3],  # too short
        [0.1, 0.2, 0.3, 0.4, 0.5],  # too long
        ["a", 0.2, 0.3, 0.4],  # non-numeric
    ],
)
def test_request_rejects_invalid_roi(roi):
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], roi_xyxyn=roi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_schemas.py -v -k roi`
Expected: the `defaults_to_none` / `accepts_*` tests FAIL (`roi_xyxyn` unknown attribute); wrong-shape cases may incidentally pass — that's fine at this stage

- [ ] **Step 3: Implement the field**

In `api/src/temporal_model/api/schemas.py`, add to `PredictRequest` (after `bucket`):

```python
    # Optional region of interest as normalized corners
    # (x_min, y_min, x_max, y_max) — ultralytics xyxyn convention, suffixed to
    # disambiguate from the xywhn bboxes in responses. Tubes with no real
    # detection intersecting it are dropped before scoring (see
    # docs/specs/2026-06-10-api-roi-design.md).
    roi_xyxyn: tuple[float, float, float, float] | None = None
```

and the validator (after `_validate_bucket`):

```python
    @field_validator("roi_xyxyn")
    @classmethod
    def _validate_roi(
        cls, v: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if v is None:
            return v
        x_min, y_min, x_max, y_max = v
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("roi_xyxyn coordinates must be in [0, 1]")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("roi_xyxyn requires x_min < x_max and y_min < y_max")
        return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_schemas.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): roi_xyxyn optional request field with [0,1] corner validation"
```

---

### Task 5: `num_tubes_outside_roi` in the verbose response

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py:77-141` (`Preprocessing` DTO + `_to_details`)
- Modify: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_schemas.py` (reuses the existing `_details`/`_tube` helpers in that file):

```python
def test_verbose_details_map_num_tubes_outside_roi():
    details = _details([_tube(1, 0.9)])
    details["tubes"]["num_outside_roi"] = 3
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3, details=details)
    resp = to_response(out, name="m", version="1", calibrated=True, verbose=True)
    assert resp.details.preprocessing.num_tubes_outside_roi == 3


def test_verbose_details_num_tubes_outside_roi_defaults_to_zero():
    # Core dumps from before the ROI feature lack the key.
    details = _details([_tube(1, 0.9)])
    assert "num_outside_roi" not in details["tubes"]
    out = SimpleNamespace(is_positive=True, trigger_frame_index=3, details=details)
    resp = to_response(out, name="m", version="1", calibrated=True, verbose=True)
    assert resp.details.preprocessing.num_tubes_outside_roi == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_schemas.py -v -k outside_roi`
Expected: FAIL — `Preprocessing` has no attribute `num_tubes_outside_roi`

- [ ] **Step 3: Implement the mapping**

In `api/src/temporal_model/api/schemas.py`:

```python
class Preprocessing(BaseModel):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]
    num_tube_candidates: int
    num_tubes_outside_roi: int = 0
```

and in `_to_details`, extend the `Preprocessing(...)` construction:

```python
        preprocessing=Preprocessing(
            num_frames_input=pre["num_frames_input"],
            num_truncated=pre["num_truncated"],
            padded_frame_indices=pre["padded_frame_indices"],
            num_tube_candidates=tubes_block["num_candidates"],
            num_tubes_outside_roi=tubes_block.get("num_outside_roi", 0),
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_schemas.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): expose num_tubes_outside_roi in verbose details"
```

---

### Task 6: thread `roi` through `ModelRunner`

**Files:**
- Modify: `api/src/temporal_model/api/model_runner.py:120-158`
- Modify: `api/tests/test_model_runner.py`

- [ ] **Step 1: Update the fakes and write the failing tests**

In `api/tests/test_model_runner.py`, `ModelRunner` will pass `roi=` to `model.predict`, so both fakes must accept it. Update `_OrchestrationModel`:

```python
    def __init__(self):
        self.detect_calls: list[list[str]] = []
        self.predict_calls: list[set[str]] = []
        self.roi_calls: list[tuple | None] = []
```

```python
    def predict(self, frames, *, frame_detections=None, roi=None, timer=None):
        self.predict_calls.append(set(frame_detections or {}))
        self.roi_calls.append(roi)
        return SimpleNamespace(frame_ids=[f.frame_id for f in frames])
```

and `_StubModel.predict`:

```python
    def predict(self, frames, *, frame_detections=None, roi=None, timer=None):
        self.predict_timer = timer
        if timer is not None:
            with timer.stage("classifier"):
                pass
        return SimpleNamespace(is_positive=False, trigger_frame_index=None, details={})
```

Add the tests (after `test_predict_resolves_all_detections_for_model`):

```python
def test_predict_threads_roi_to_model():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"], roi=(0.1, 0.2, 0.3, 0.4)))
    assert model.roi_calls[-1] == (0.1, 0.2, 0.3, 0.4)


def test_predict_roi_defaults_to_none():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"]))
    assert model.roi_calls[-1] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_model_runner.py -v`
Expected: the two new tests FAIL — `ModelRunner.predict() got an unexpected keyword argument 'roi'`; existing tests still PASS

- [ ] **Step 3: Implement the threading**

In `api/src/temporal_model/api/model_runner.py`:

```python
    async def predict(
        self,
        frame_paths: list[Path],
        *,
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve detections (cache + detect misses) then run the model.

        The whole orchestration runs in a worker thread under the lock, so the
        cache is accessed by one prediction at a time. When ``timer``/``profile``
        are supplied, the ``detector`` stage is timed and cache counts recorded.
        ``roi`` is passed through to the core model untouched — the cache stays
        full-frame (see the invariant in the ROI spec).
        """
        async with self._lock:
            return await run_in_threadpool(
                self._predict_sync, frame_paths, roi, timer, profile
            )

    def _predict_sync(
        self,
        frame_paths: list[Path],
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
```

and the model call inside `_predict_sync` becomes:

```python
        out = self._model.predict(
            frames, frame_detections=resolved, roi=roi, timer=timer
        )
```

(Everything else in `_predict_sync` — cache resolution, `put()` of full-frame detections — is untouched. Do NOT filter detections by ROI here; that would poison the shared cache.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_model_runner.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/model_runner.py api/tests/test_model_runner.py
git commit -m "feat(api): thread roi through ModelRunner to core predict"
```

---

### Task 7: wire `app.py` and cover the HTTP path

**Files:**
- Modify: `api/src/temporal_model/api/app.py:146`
- Modify: `api/tests/test_app.py`
- Modify: `api/README.md:11-12`

- [ ] **Step 1: Update `FakeRunner` and write the failing tests**

In `api/tests/test_app.py`, update `FakeRunner` to record the roi:

```python
class FakeRunner:
    name = "bbox-tube-vit-dinov2"
    version = "1.2.0"
    calibrated = True
    threshold_overridden = False
    packaged_threshold = None

    def __init__(self, output=None, error=None):
        self._output = output
        self._error = error
        self.roi = "UNSET"  # sentinel: distinguishes "not passed" from None

    async def predict(self, paths, *, roi=None, timer=None, profile=None):
        self.roi = roi
        if self._error:
            raise self._error
        if timer is not None:
            with timer.stage("detector"):
                pass
        if profile is not None:
            profile.update(n_frames=len(paths), cache_hits=0, cache_misses=len(paths))
        return self._output
```

Add the tests (near the other `/predict` tests):

```python
def test_predict_passes_roi_to_runner(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "roi_xyxyn": [0.1, 0.2, 0.3, 0.4]}
    )
    assert r.status_code == 200
    assert client.app.state.runner.roi == (0.1, 0.2, 0.3, 0.4)


def test_predict_without_roi_passes_none(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert client.app.state.runner.roi is None


def test_predict_invalid_roi_is_400(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "roi_xyxyn": [0.3, 0.2, 0.1, 0.4]}
    )
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_request"
    assert "roi_xyxyn" in body["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_app.py -v -k roi`
Expected: `test_predict_passes_roi_to_runner` FAILS (runner.roi stays `None` — app.py doesn't pass it yet); the invalid-roi test already PASSES (validation landed in Task 4)

- [ ] **Step 3: Wire the app**

In `api/src/temporal_model/api/app.py` line 146, change:

```python
            out = await runner.predict(paths, timer=timer, profile=profile)
```

to:

```python
            out = await runner.predict(
                paths, roi=body.roi_xyxyn, timer=timer, profile=profile
            )
```

- [ ] **Step 4: Run the API suite**

Run: `cd api && uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Update `api/README.md`**

Line 11-12, document the new field:

```markdown
- `POST /predict` — body `{ "frames": ["<s3-key>", ...], "bucket": "<name>",
  "roi_xyxyn": [x_min, y_min, x_max, y_max] }`
  (ordered S3 keys; `bucket` optional, falls back to `S3_BUCKET`;
  `roi_xyxyn` optional normalized region of interest — tubes with no real
  detection intersecting it are dropped before scoring);
```

- [ ] **Step 6: Commit**

```bash
git add api/src/temporal_model/api/app.py api/tests/test_app.py api/README.md
git commit -m "feat(api): accept roi_xyxyn on /predict and scope the verdict to it"
```

---

### Task 8: full verification

- [ ] **Step 1: Run both package suites**

Run: `make -C core test && make -C api test`
Expected: all PASS

- [ ] **Step 2: Lint**

Run: `make -C core lint && make -C api lint`
Expected: clean; fix anything flagged in the files this plan touched, re-run, and amend the relevant commit or add a `style:` commit

- [ ] **Step 3: Spec cross-check**

Re-read `docs/specs/2026-06-10-api-roi-design.md` section by section and confirm each requirement maps to landed code: request field + validation (Task 4), tube-level intersection rule incl. gap exclusion (Task 1), filter before scoring (Task 3), `num_outside_roi` core + `num_tubes_outside_roi` API (Tasks 2, 5), pass-through plumbing + cache invariant (Tasks 6, 7), `roi` omitted → byte-identical behavior (Task 3 baseline tests + Task 7 none-test).

- [ ] **Step 4 (optional, manual): sanity-check on the real sequence**

Not a committed test (scratch data is not in the repo). Serve the packaged model locally and POST `scratch/annot_seq_9711` frames once with an ROI around the x≈0.38 cluster (e.g. `[0.30, 0.35, 0.50, 0.55]`) and once around x≈0.14 (e.g. `[0.05, 0.35, 0.25, 0.55]`) with `?verbose=true`: the kept tubes must differ and `num_tubes_outside_roi` must be ≥ 1 in each case.
