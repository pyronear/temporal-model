# Caller-Supplied Detections on `/predict` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `/predict` callers supply per-frame detection boxes (from the RPi edge detector) so the API skips its bundled YOLO pass while tube building, ROI filtering, cropping, and classification run unchanged.

**Architecture:** A new optional `detections` field on `PredictRequest` (index-aligned list of per-frame box lists, `xyxyn` + `confidence`), validated at the HTTP boundary. `ModelRunner._predict_sync` converts supplied boxes to internal `FrameDetections` and feeds the existing `model.predict(frame_detections=...)` injection seam, skipping `detect()` and the detection cache entirely. Core is untouched. Spec: `docs/specs/2026-06-11-api-supplied-detections-design.md`.

**Tech Stack:** FastAPI + Pydantic v2 (`api/` package), pytest, `uv` (run tests from `api/`: `uv run pytest tests/ -v`).

**Branch:** work on `arthur/feat-api-thread-bboxes`. Commit messages: conventional commits, NO co-author trailers.

---

### Task 1: Request schema — `SuppliedDetection` + `detections` field

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py` (imports at line 11, `PredictRequest` at lines 22–69)
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_schemas.py` (existing imports at top already provide `pytest`, `ValidationError`, `PredictRequest`):

```python
def test_request_detections_default_to_none():
    assert PredictRequest(frames=["a.jpg"]).detections is None


def test_request_accepts_per_frame_detections():
    req = PredictRequest(
        frames=["a.jpg", "b.jpg"],
        detections=[
            [{"xyxyn": [0.1, 0.2, 0.3, 0.4], "confidence": 0.7}],
            [],
        ],
    )
    assert req.detections[0][0].xyxyn == (0.1, 0.2, 0.3, 0.4)
    assert req.detections[0][0].confidence == 0.7
    assert req.detections[1] == []


@pytest.mark.parametrize("entries", [[], [[]], [[], [], []]])
def test_request_rejects_detections_length_mismatch(entries):
    # frames has 2 keys; 0, 1 and 3 detection entries must all fail.
    with pytest.raises(ValidationError, match="one entry per frame"):
        PredictRequest(frames=["a.jpg", "b.jpg"], detections=entries)


def test_request_rejects_null_frame_entry():
    # "no detections" must be an explicit [], never null.
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], detections=[None])


@pytest.mark.parametrize(
    "box",
    [
        {"xyxyn": [-0.1, 0.2, 0.3, 0.4], "confidence": 0.5},  # coord < 0
        {"xyxyn": [0.1, 0.2, 0.3, 1.4], "confidence": 0.5},  # coord > 1
        {"xyxyn": [0.3, 0.2, 0.1, 0.4], "confidence": 0.5},  # x_min >= x_max
        {"xyxyn": [0.1, 0.4, 0.3, 0.4], "confidence": 0.5},  # y_min >= y_max
        {"xyxyn": [0.1, 0.2, 0.3], "confidence": 0.5},  # too short
        {"xyxyn": [0.1, 0.2, 0.3, 0.4, 0.5], "confidence": 0.5},  # too long
        {"xyxyn": ["a", 0.2, 0.3, 0.4], "confidence": 0.5},  # non-numeric
        {"xyxyn": [0.1, 0.2, 0.3, 0.4], "confidence": 1.5},  # confidence > 1
        {"xyxyn": [0.1, 0.2, 0.3, 0.4], "confidence": -0.1},  # confidence < 0
        {"xyxyn": [0.1, 0.2, 0.3, 0.4]},  # missing confidence
        {"confidence": 0.5},  # missing xyxyn
        "not-an-object",  # wrong type entirely
    ],
)
def test_request_rejects_malformed_detection(box):
    with pytest.raises(ValidationError):
        PredictRequest(frames=["a.jpg"], detections=[[box]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_schemas.py -v -k detection`
Expected: FAIL — `test_request_detections_default_to_none` errors with `AttributeError: 'PredictRequest' object has no attribute 'detections'`; the rejection tests fail because no `ValidationError` is raised (unknown fields are ignored).

- [ ] **Step 3: Implement the schema**

In `api/src/temporal_model/api/schemas.py`:

3a. Extend the pydantic import (line 11):

```python
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
```

3b. Add `SuppliedDetection` immediately above `class PredictRequest` (after the `_BUCKET_RE` block):

```python
class SuppliedDetection(BaseModel):
    """One caller-supplied detection box (normalized xyxyn corners).

    Geometry rules match ``roi_xyxyn``. Checked inline rather than via the
    core ``validate_roi`` helper so the error message names the detection
    field, not "roi".
    """

    xyxyn: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("xyxyn")
    @classmethod
    def _validate_xyxyn(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = v
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("xyxyn coordinates must be in [0, 1]")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("xyxyn requires x_min < x_max and y_min < y_max")
        return v
```

3c. Add the field to `PredictRequest` directly under `roi_xyxyn` (line 33):

```python
    # Optional caller-supplied detections, one list per frame, index-aligned
    # with `frames` ([] = that frame's detector saw nothing — never null).
    # When set, the bundled YOLO and its cache are bypassed entirely and tubes
    # are built from these boxes (see
    # docs/specs/2026-06-11-api-supplied-detections-design.md).
    detections: list[list[SuppliedDetection]] | None = None
```

3d. Add the cross-field length check after `_validate_roi` (line 69):

```python
    @model_validator(mode="after")
    def _detections_match_frames(self) -> "PredictRequest":
        if self.detections is not None and len(self.detections) != len(self.frames):
            raise ValueError(
                "detections must have exactly one entry per frame "
                f"(got {len(self.detections)} entries for {len(self.frames)} frames)"
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_schemas.py -v`
Expected: all PASS (new tests and pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): accept per-frame supplied detections in PredictRequest"
```

---

### Task 2: Runner bypass — skip detector and cache when detections are supplied

**Files:**
- Modify: `api/src/temporal_model/api/model_runner.py` (imports ~line 19, `predict` at lines 120–139, `_predict_sync` at lines 141–177)
- Test: `api/tests/test_model_runner.py` (`_OrchestrationModel` at lines 117–143)

- [ ] **Step 1: Extend the orchestration fake to record full detections**

In `api/tests/test_model_runner.py`, update `_OrchestrationModel` so `detect()` returns a real box (clean binary-fraction floats so dataclass equality is exact) and `predict()` records the full `frame_detections` dict:

```python
class _OrchestrationModel:
    """Fake core model recording how detection is invoked across calls."""

    def __init__(self):
        self.detect_calls: list[list[str]] = []
        self.predict_calls: list[set[str]] = []
        self.roi_calls: list[tuple | None] = []
        self.frame_detections_calls: list[dict] = []

    def load_sequence(self, paths):
        return [
            Frame(frame_id=Path(p).stem, image_path=Path(p), timestamp=None)
            for p in paths
        ]

    def detect(self, frames):
        self.detect_calls.append([f.frame_id for f in frames])
        return [
            FrameDetections(
                frame_idx=i,
                frame_id=f.frame_id,
                timestamp=None,
                detections=[
                    Detection(
                        class_id=0, cx=0.5, cy=0.5, w=0.5, h=0.5, confidence=0.75
                    )
                ],
            )
            for i, f in enumerate(frames)
        ]

    def predict(self, frames, *, frame_detections=None, roi=None, timer=None):
        self.predict_calls.append(set(frame_detections or {}))
        self.roi_calls.append(roi)
        self.frame_detections_calls.append(frame_detections or {})
        return SimpleNamespace(frame_ids=[f.frame_id for f in frames])
```

`Detection` needs importing at the top of the file alongside the existing `Frame`/`FrameDetections` import (check the import block at the top; it already imports from `temporal_model.core.types`):

```python
from temporal_model.core.types import Detection, Frame, FrameDetections
```

Existing tests only inspect `detect_calls` ids and `predict_calls` key sets, so the richer `detect()` return changes nothing for them.

- [ ] **Step 2: Write the failing tests**

Append to `api/tests/test_model_runner.py`. Add `pytest` and `SuppliedDetection` imports at the top if missing:

```python
import pytest

from temporal_model.api.schemas import SuppliedDetection
```

Tests:

```python
def _supplied_box():
    return SuppliedDetection(xyxyn=(0.1, 0.2, 0.5, 0.8), confidence=0.7)


def test_predict_with_supplied_detections_skips_detect():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(
        runner.predict(
            ["c/x_00.jpg", "c/x_01.jpg"], detections=[[_supplied_box()], []]
        )
    )

    assert model.detect_calls == []
    assert model.predict_calls[-1] == {"x_00", "x_01"}


def test_predict_supplied_detections_converted_to_xywhn():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"], detections=[[_supplied_box()]]))

    fd = model.frame_detections_calls[-1]["x_00"]
    assert fd.frame_idx == 0
    [det] = fd.detections
    assert (det.cx, det.cy, det.w, det.h) == pytest.approx((0.3, 0.5, 0.4, 0.6))
    assert det.confidence == 0.7
    assert det.class_id == 0


def test_predict_supplied_empty_frame_has_no_detections():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"], detections=[[]]))

    assert model.frame_detections_calls[-1]["x_00"].detections == []


def test_predict_supplied_detections_do_not_enter_cache():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=4096
    )
    asyncio.run(runner.predict(["c/x_00.jpg"], detections=[[_supplied_box()]]))
    asyncio.run(runner.predict(["c/x_00.jpg"]))

    # The supplied run wrote nothing: the plain run must re-detect the frame.
    assert model.detect_calls == [["x_00"]]


def test_predict_supplied_detections_ignore_cached_entries():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=4096
    )
    asyncio.run(runner.predict(["c/x_00.jpg"]))  # warms cache (confidence 0.75)
    asyncio.run(runner.predict(["c/x_00.jpg"], detections=[[_supplied_box()]]))

    # The supplied run used the supplied box, not the cached detector output.
    [det] = model.frame_detections_calls[-1]["x_00"].detections
    assert det.confidence == 0.7


def test_predict_supplied_detections_profile_counters():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    profile: dict = {}
    asyncio.run(
        runner.predict(
            ["c/x_00.jpg"], detections=[[_supplied_box()]], profile=profile
        )
    )

    assert profile == {"n_frames": 1, "cache_hits": 0, "cache_misses": 0}


def test_predict_supplied_detections_threads_roi():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(
        runner.predict(
            ["c/x_00.jpg"], detections=[[_supplied_box()]], roi=(0.1, 0.2, 0.3, 0.4)
        )
    )

    assert model.roi_calls[-1] == (0.1, 0.2, 0.3, 0.4)


def test_predict_supplied_matches_detector_path():
    # Supplying the exact box the detector would produce (xywhn 0.5/0.5/0.5/0.5
    # == xyxyn 0.25..0.75) hands the model identical FrameDetections.
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    paths = ["c/x_00.jpg", "c/x_01.jpg"]

    asyncio.run(runner.predict(paths))
    detector_fds = model.frame_detections_calls[-1]

    equivalent = SuppliedDetection(xyxyn=(0.25, 0.25, 0.75, 0.75), confidence=0.75)
    asyncio.run(runner.predict(paths, detections=[[equivalent], [equivalent]]))
    supplied_fds = model.frame_detections_calls[-1]

    assert supplied_fds == detector_fds
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_model_runner.py -v -k supplied`
Expected: FAIL — `TypeError: ModelRunner.predict() got an unexpected keyword argument 'detections'`.

- [ ] **Step 4: Implement the runner bypass**

In `api/src/temporal_model/api/model_runner.py`:

4a. Add imports at the top (after the existing `from temporal_model.core.stage_timer import ...` line):

```python
from temporal_model.core.types import Detection, FrameDetections

from .detection_cache import DetectionCache
from .schemas import SuppliedDetection
```

(`.detection_cache` is already imported; only add the other two lines, keeping import order ruff-clean.)

4b. Add a module-level helper above `class ModelRunner`:

```python
def _supplied_frame_detections(
    frames: list[Any], detections: list[list[SuppliedDetection]]
) -> dict[str, FrameDetections]:
    """Convert caller-supplied xyxyn boxes to per-frame ``FrameDetections``.

    ``detections`` is index-aligned with ``frames`` (lengths validated at the
    HTTP boundary; ``strict=True`` is a safety net). Boxes arrive as
    normalized corners and become center-based xywhn ``Detection``s; supplied
    boxes are smoke by definition (``class_id=0``).
    """
    resolved: dict[str, FrameDetections] = {}
    for idx, (frame, boxes) in enumerate(zip(frames, detections, strict=True)):
        resolved[frame.frame_id] = FrameDetections(
            frame_idx=idx,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            detections=[
                Detection(
                    class_id=0,
                    cx=(b.xyxyn[0] + b.xyxyn[2]) / 2.0,
                    cy=(b.xyxyn[1] + b.xyxyn[3]) / 2.0,
                    w=b.xyxyn[2] - b.xyxyn[0],
                    h=b.xyxyn[3] - b.xyxyn[1],
                    confidence=b.confidence,
                )
                for b in boxes
            ],
        )
    return resolved
```

4c. Thread the parameter through `predict` (signature + docstring + threadpool call):

```python
    async def predict(
        self,
        frame_paths: list[Path],
        *,
        roi: tuple[float, float, float, float] | None = None,
        detections: list[list[SuppliedDetection]] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve detections (cache + detect misses) then run the model.

        The whole orchestration runs in a worker thread under the lock, so the
        cache is accessed by one prediction at a time. When ``timer``/``profile``
        are supplied, the ``detector`` stage is timed and cache counts recorded.
        ``roi`` is passed through to the core model untouched — the cache stays
        full-frame (see the invariant in the ROI spec). When ``detections`` is
        supplied (index-aligned per-frame boxes from the caller's own
        detector), the bundled detector and its cache are bypassed entirely:
        no read, no write, no ``detector`` stage.
        """
        async with self._lock:
            return await run_in_threadpool(
                self._predict_sync, frame_paths, roi, detections, timer, profile
            )
```

4d. Branch in `_predict_sync` right after `load_sequence`:

```python
    def _predict_sync(
        self,
        frame_paths: list[Path],
        roi: tuple[float, float, float, float] | None = None,
        detections: list[list[SuppliedDetection]] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        started = time.perf_counter()
        frames = self._model.load_sequence(frame_paths)
        if detections is not None:
            out = self._model.predict(
                frames,
                frame_detections=_supplied_frame_detections(frames, detections),
                roi=roi,
                timer=timer,
            )
            if profile is not None:
                profile["n_frames"] = len(frames)
                profile["cache_hits"] = 0
                profile["cache_misses"] = 0
            logger.info(
                "predict: supplied detections, seq_len=%d, %.0fms",
                len(frames),
                (time.perf_counter() - started) * 1000.0,
            )
            return out
        resolved: dict[str, Any] = {}
        ...  # existing detector-path body continues unchanged from here
```

(The `...` is the existing code from `resolved: dict[str, Any] = {}` onward — do not modify it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_model_runner.py -v`
Expected: all PASS (new and pre-existing).

- [ ] **Step 6: Commit**

```bash
git add api/src/temporal_model/api/model_runner.py api/tests/test_model_runner.py
git commit -m "feat(api): bypass detector and cache when detections are supplied"
```

---

### Task 3: App threading + `detections_source` provenance in verbose details

**Files:**
- Modify: `api/src/temporal_model/api/app.py` (predict handler, lines 146–169)
- Modify: `api/src/temporal_model/api/schemas.py` (`Preprocessing` ~line 100, `_to_details` ~line 142, `to_response` ~line 171)
- Test: `api/tests/test_app.py` (`FakeRunner` at lines 58–79), `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

1a. Append to `api/tests/test_schemas.py`:

```python
def test_verbose_details_detections_source_request():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(
        out,
        name="m",
        version="1",
        calibrated=True,
        verbose=True,
        detections_source="request",
    )
    assert resp.details.preprocessing.detections_source == "request"


def test_verbose_details_detections_source_defaults_to_detector():
    out = SimpleNamespace(
        is_positive=False, trigger_frame_index=None, details=_details([])
    )
    resp = to_response(out, name="m", version="1", calibrated=True, verbose=True)
    assert resp.details.preprocessing.detections_source == "detector"
```

1b. In `api/tests/test_app.py`, update `FakeRunner` to accept and record the new kwarg:

```python
    def __init__(self, output=None, error=None):
        self._output = output
        self._error = error
        self.roi = None
        self.detections = None

    async def predict(
        self, paths, *, roi=None, detections=None, timer=None, profile=None
    ):
        self.roi = roi
        self.detections = detections
        if self._error:
            raise self._error
        if timer is not None:
            with timer.stage("detector"):
                pass
        if profile is not None:
            profile.update(n_frames=len(paths), cache_hits=0, cache_misses=len(paths))
        return self._output
```

1c. Append endpoint tests to `api/tests/test_app.py` (`KEYS` has exactly 2 frames):

```python
def test_predict_passes_detections_to_runner(client):
    r = client.post(
        "/predict",
        json={
            "frames": KEYS,
            "detections": [
                [{"xyxyn": [0.1, 0.2, 0.3, 0.4], "confidence": 0.6}],
                [],
            ],
        },
    )
    assert r.status_code == 200
    sent = client.app.state.runner.detections
    assert sent[0][0].xyxyn == (0.1, 0.2, 0.3, 0.4)
    assert sent[0][0].confidence == 0.6
    assert sent[1] == []


def test_predict_without_detections_passes_none(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert client.app.state.runner.detections is None


def test_predict_detections_length_mismatch_is_400(client):
    r = client.post("/predict", json={"frames": KEYS, "detections": [[]]})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_request"
    assert "one entry per frame" in body["detail"]


def test_predict_malformed_detection_is_400(client):
    r = client.post(
        "/predict",
        json={
            "frames": KEYS,
            "detections": [
                [{"xyxyn": [0.3, 0.2, 0.1, 0.4], "confidence": 0.6}],
                [],
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_detections_compose_with_roi(client):
    r = client.post(
        "/predict",
        json={
            "frames": KEYS,
            "detections": [[], []],
            "roi_xyxyn": [0.0, 0.0, 1.0, 1.0],
        },
    )
    assert r.status_code == 200
    assert client.app.state.runner.roi == (0.0, 0.0, 1.0, 1.0)
    assert client.app.state.runner.detections == [[], []]


def test_predict_verbose_detections_source_request(client):
    r = client.post(
        "/predict?verbose=true", json={"frames": KEYS, "detections": [[], []]}
    )
    assert r.status_code == 200
    assert r.json()["details"]["preprocessing"]["detections_source"] == "request"


def test_predict_verbose_detections_source_detector(client):
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json()["details"]["preprocessing"]["detections_source"] == "detector"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/test_app.py tests/test_schemas.py -v -k "detections_source or detections"`
Expected: FAIL — `to_response() got an unexpected keyword argument 'detections_source'`; endpoint tests fail with `detections_source` missing from the verbose payload and `FakeRunner.detections` never set by the app.

- [ ] **Step 3: Implement**

3a. In `api/src/temporal_model/api/schemas.py`, add the field to `Preprocessing`:

```python
class Preprocessing(BaseModel):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]
    num_tube_candidates: int
    num_tubes_outside_roi: int
    # Provenance: "request" when the caller supplied the detections (bundled
    # detector bypassed), "detector" when the bundled YOLO produced them.
    detections_source: Literal["request", "detector"]
```

3b. Thread it through `_to_details` (add parameter, pass into `Preprocessing(...)`):

```python
def _to_details(
    details: dict[str, Any],
    *,
    threshold_overridden: bool,
    packaged_threshold: float | None,
    detections_source: Literal["request", "detector"],
    profiling: dict[str, Any] | None = None,
) -> Details:
```

and inside the `Preprocessing(` call add `detections_source=detections_source,`.

3c. Thread it through `to_response` (default keeps every existing caller working):

```python
def to_response(
    out: Any,
    *,
    name: str,
    version: str | None,
    calibrated: bool,
    verbose: bool,
    threshold_overridden: bool = False,
    packaged_threshold: float | None = None,
    detections_source: Literal["request", "detector"] = "detector",
    profiling: dict[str, Any] | None = None,
) -> PredictResponse:
```

and in the `verbose` branch pass `detections_source=detections_source,` to `_to_details`.

3d. In `api/src/temporal_model/api/app.py`, update the two call sites in `predict`:

```python
            out = await runner.predict(
                paths,
                roi=body.roi_xyxyn,
                detections=body.detections,
                timer=timer,
                profile=profile,
            )
```

```python
            return to_response(
                out,
                name=runner.name,
                version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                detections_source=(
                    "request" if body.detections is not None else "detector"
                ),
                profiling=profiling,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_app.py tests/test_schemas.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/app.py api/src/temporal_model/api/schemas.py api/tests/test_app.py api/tests/test_schemas.py
git commit -m "feat(api): thread supplied detections through /predict with provenance"
```

---

### Task 4: README, lint, full suite

**Files:**
- Modify: `api/README.md` (the `POST /predict` bullet, lines 11–16)

- [ ] **Step 1: Update the endpoint documentation**

Replace the `POST /predict` bullet in `api/README.md` with:

```markdown
- `POST /predict` — body `{ "frames": ["<s3-key>", ...], "bucket": "<name>",
  "roi_xyxyn": [x_min, y_min, x_max, y_max],
  "detections": [[{"xyxyn": [...], "confidence": 0.6}], []] }`
  (ordered S3 keys; `bucket` optional, falls back to `S3_BUCKET`;
  `roi_xyxyn` optional normalized region of interest — tubes with no real
  detection intersecting it are dropped before scoring;
  `detections` optional caller-supplied boxes, one list per frame
  index-aligned with `frames`, `[]` = that frame's detector saw nothing —
  skips the bundled YOLO and its cache entirely, tubes are built from the
  supplied boxes);
  returns `{ is_smoke, probability, model }` (`probability` = max kept-tube
  calibrated probability, `null` if uncalibrated).
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks).
```

- [ ] **Step 2: Lint and run the full API suite**

Run: `make -C api lint && make -C api test`
Expected: lint clean, all tests PASS.

- [ ] **Step 3: Run the core suite (regression — core is meant to be untouched)**

Run: `git status --short core/` (expect no modifications) and `make -C core test`
Expected: no core diffs; all core tests PASS.

- [ ] **Step 4: Commit**

```bash
git add api/README.md
git commit -m "docs(api): document the detections field on /predict"
```
