# `?compute_trigger=true` on `/predict` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `?compute_trigger=true` query param to `/predict` that runs the first-crossing search and returns time-to-detection fields, leaving the default response byte-identical (issue #26).

**Architecture:** Pure plumbing in the `api` package — the core model already accepts `compute_trigger` and emits `trigger_frame_index` / `trigger_tube_id` / per-tube `first_crossing_frame`. The flag is threaded `app.py` route → `ModelRunner.predict` → `_predict_sync` → `model.predict`, and the response mapper gates the new optional DTO fields on it. Serialization relies on the route's existing `response_model_exclude_unset=True`: fields stay *unset* (omitted) unless the flag is on; when on, `null` means "searched, no crossing".

**Tech Stack:** FastAPI + pydantic v2, pytest, uv. All commands run from `api/`.

**Spec:** `docs/specs/2026-06-11-api-compute-trigger-design.md`

---

### Task 1: Thread `compute_trigger` through `ModelRunner`

**Files:**
- Modify: `api/src/temporal_model/api/model_runner.py:120-177`
- Test: `api/tests/test_model_runner.py`

- [ ] **Step 1: Write the failing tests**

In `api/tests/test_model_runner.py`, extend the `_OrchestrationModel` fake (line 117) to record the flag, mirroring the existing `roi_calls` pattern. In `__init__` add:

```python
        self.trigger_calls: list[bool] = []
```

Change its `predict` method (line 140) to accept and record the kwarg:

```python
    def predict(
        self,
        frames,
        *,
        frame_detections=None,
        roi=None,
        timer=None,
        compute_trigger=False,
    ):
        self.predict_calls.append(set(frame_detections or {}))
        self.roi_calls.append(roi)
        self.trigger_calls.append(compute_trigger)
        return SimpleNamespace(frame_ids=[f.frame_id for f in frames])
```

Add two tests right after `test_predict_roi_defaults_to_none` (line 167):

```python
def test_predict_threads_compute_trigger_to_model():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"], compute_trigger=True))
    assert model.trigger_calls[-1] is True


def test_predict_compute_trigger_defaults_to_false():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"]))
    assert model.trigger_calls[-1] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd api && uv run pytest tests/test_model_runner.py -q -k compute_trigger`
Expected: 2 FAILED — `TypeError: ModelRunner.predict() got an unexpected keyword argument 'compute_trigger'`

- [ ] **Step 3: Implement the threading**

In `api/src/temporal_model/api/model_runner.py`, add the keyword to `predict` (line 120):

```python
    async def predict(
        self,
        frame_paths: list[Path],
        *,
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
        compute_trigger: bool = False,
    ) -> Any:
```

and forward it (line 137):

```python
        async with self._lock:
            return await run_in_threadpool(
                self._predict_sync, frame_paths, roi, timer, profile, compute_trigger
            )
```

Add it to `_predict_sync` (line 141):

```python
    def _predict_sync(
        self,
        frame_paths: list[Path],
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
        compute_trigger: bool = False,
    ) -> Any:
```

and pass it to the core model (line 162):

```python
        out = self._model.predict(
            frames,
            frame_detections=resolved,
            roi=roi,
            timer=timer,
            compute_trigger=compute_trigger,
        )
```

- [ ] **Step 4: Run the test file to verify it passes**

Run: `cd api && uv run pytest tests/test_model_runner.py -q`
Expected: all pass (existing orchestration tests still green — the fake's new kwarg has a default).

- [ ] **Step 5: Commit**

```bash
git add api/tests/test_model_runner.py api/src/temporal_model/api/model_runner.py
git commit -m "feat(api): thread compute_trigger through ModelRunner"
```

---

### Task 2: Gate trigger fields in the response DTOs

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py` (`Tube`, `Decision`, `PredictResponse`, `_to_details`, `to_response`)
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/test_schemas.py`. The existing `_details()` / `_tube()` helpers (lines 9-42) already carry `trigger_tube_id` and `first_crossing_frame` keys — exactly what core emits — so the tests prove gating is by flag, not by data presence.

```python
def test_compute_trigger_sets_top_level_trigger_frame_index():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=False,
        compute_trigger=True,
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert dumped["trigger_frame_index"] == 3
    assert "details" not in dumped


def test_compute_trigger_no_crossing_is_explicit_null():
    # Searched but nothing crossed: the key is present with an explicit null.
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None, details=_details([]))
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=False,
        compute_trigger=True,
    )
    dumped = resp.model_dump(exclude_unset=True)
    assert "trigger_frame_index" in dumped
    assert dumped["trigger_frame_index"] is None


def test_default_omits_trigger_frame_index():
    # Even when the core output carries a trigger, the flag gates exposure.
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=False
    )
    assert "trigger_frame_index" not in resp.model_dump(exclude_unset=True)


def test_compute_trigger_verbose_adds_trigger_details():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out,
        api_version=None,
        model_version="1.2.0",
        calibrated=True,
        verbose=True,
        compute_trigger=True,
    )
    details = resp.model_dump(exclude_unset=True)["details"]
    assert details["decision"]["trigger_tube_id"] == 7
    assert details["tubes"][0]["first_crossing_frame"] == 3


def test_verbose_without_compute_trigger_omits_trigger_details():
    out = SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details([_tube(7, 0.98)])
    )
    resp = to_response(
        out, api_version=None, model_version="1.2.0", calibrated=True, verbose=True
    )
    details = resp.model_dump(exclude_unset=True)["details"]
    assert "trigger_tube_id" not in details["decision"]
    assert "first_crossing_frame" not in details["tubes"][0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd api && uv run pytest tests/test_schemas.py -q -k trigger`
Expected: FAILED — `TypeError: to_response() got an unexpected keyword argument 'compute_trigger'` (the two no-flag tests may pass already; the three flag tests must fail).

- [ ] **Step 3: Implement the DTO gating**

In `api/src/temporal_model/api/schemas.py`:

Add the optional field to `Tube` (line 84), after `probability` to match core's `KeptTube` order:

```python
class Tube(BaseModel):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None = None
    entries: list[FrameEntry]
```

Add the optional field to `Decision` (line 93):

```python
class Decision(BaseModel):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float
    threshold_overridden: bool = False
    packaged_threshold: float | None = None
    trigger_tube_id: int | None = None
```

Add the optional field to `PredictResponse` (line 127):

```python
class PredictResponse(BaseModel):
    is_smoke: bool
    probability: float | None
    trigger_frame_index: int | None = None
    version: Version
    details: Details | None = None
```

Rewrite `_to_details` (line 147) to gate the trigger keys. Core always emits them (as `None` on the fast path); dropping them when the flag is off keeps the DTO fields *unset* so `exclude_unset` omits them:

```python
def _to_details(
    details: dict[str, Any],
    *,
    threshold_overridden: bool,
    packaged_threshold: float | None,
    profiling: dict[str, Any] | None = None,
    compute_trigger: bool = False,
) -> Details:
    tubes_block = details["tubes"]
    pre = details["preprocessing"]
    decision = dict(details["decision"])
    kept = tubes_block["kept"]
    if not compute_trigger:
        # Core emits these keys even on the fast path (always null there);
        # dropping them keeps the DTO fields unset so exclude_unset omits
        # them and the no-flag response is unchanged.
        decision.pop("trigger_tube_id", None)
        kept = [
            {k: v for k, v in t.items() if k != "first_crossing_frame"}
            for t in kept
        ]
    return Details(
        decision=Decision(
            **decision,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
        ),
        preprocessing=Preprocessing(
            num_frames_input=pre["num_frames_input"],
            num_truncated=pre["num_truncated"],
            padded_frame_indices=pre["padded_frame_indices"],
            num_tube_candidates=tubes_block["num_candidates"],
            # Strict like num_candidates: core (same-commit path dependency)
            # always emits the key; a silent 0 here would mask a core rename.
            num_tubes_outside_roi=tubes_block["num_outside_roi"],
        ),
        tubes=[Tube(**t) for t in kept],
        profiling=profiling,
    )
```

Extend `to_response` (line 176):

```python
def to_response(
    out: Any,
    *,
    api_version: str | None,
    model_version: str | None,
    calibrated: bool,
    verbose: bool,
    compute_trigger: bool = False,
    threshold_overridden: bool = False,
    packaged_threshold: float | None = None,
    profiling: dict[str, Any] | None = None,
) -> PredictResponse:
    """Reshape a core model output into the public response DTO."""
    kwargs: dict[str, Any] = {
        "is_smoke": out.is_positive,
        "probability": _decision_probability(out.details, calibrated),
        "version": Version(api=api_version, model=model_version),
    }
    if compute_trigger:
        # Explicit null is meaningful here: searched, no crossing found.
        kwargs["trigger_frame_index"] = out.trigger_frame_index
    if verbose:
        kwargs["details"] = _to_details(
            out.details,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
            profiling=profiling,
            compute_trigger=compute_trigger,
        )
    return PredictResponse(**kwargs)
```

Update the module docstring (lines 1-6) to mention the second flag:

```python
"""Public request/response DTOs and the mapper from the core model output.

The default response is the lean verdict; ``?verbose=true`` adds a ``details``
block and ``?compute_trigger=true`` adds the time-to-detection fields. Both
are only set when requested, so the route serializes with
``exclude_unset=True`` to omit them otherwise (while keeping explicit
``null``s).
"""
```

- [ ] **Step 4: Run the test file to verify it passes**

Run: `cd api && uv run pytest tests/test_schemas.py -q`
Expected: all pass (existing tests prove the no-flag mapping is unchanged).

- [ ] **Step 5: Commit**

```bash
git add api/tests/test_schemas.py api/src/temporal_model/api/schemas.py
git commit -m "feat(api): gate trigger fields in response DTOs behind compute_trigger"
```

---

### Task 3: Add the `compute_trigger` query param to the route

**Files:**
- Modify: `api/src/temporal_model/api/app.py:129-179`
- Test: `api/tests/test_app.py`

- [ ] **Step 1: Write the failing tests**

In `api/tests/test_app.py`, extend `FakeRunner` to accept and record the flag. In `__init__` (line 65) add:

```python
        self.compute_trigger = None
```

Change its `predict` (line 70):

```python
    async def predict(
        self, paths, *, roi=None, timer=None, profile=None, compute_trigger=False
    ):
        self.roi = roi
        self.compute_trigger = compute_trigger
        if self._error:
            raise self._error
        if timer is not None:
            with timer.stage("detector"):
                pass
        if profile is not None:
            profile.update(n_frames=len(paths), cache_hits=0, cache_misses=len(paths))
        return self._output
```

Add three tests after `test_predict_verbose_surfaces_override` (line 212). Note the fake output already carries `trigger_frame_index=3` and trigger detail keys, so `test_predict_default` (line 178, exact-equality assert) keeps guarding that the default response is unchanged.

```python
def test_predict_compute_trigger_returns_trigger_frame_index(client):
    r = client.post("/predict?compute_trigger=true", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json() == {
        "is_smoke": True,
        "probability": 0.98,
        "trigger_frame_index": 3,
        "version": {"api": None, "model": "1.2.0"},
    }
    assert client.app.state.runner.compute_trigger is True


def test_predict_default_runs_fast_path(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert "trigger_frame_index" not in r.json()
    assert client.app.state.runner.compute_trigger is False


def test_predict_compute_trigger_verbose_adds_trigger_details(client):
    r = client.post(
        "/predict?compute_trigger=true&verbose=true", json={"frames": KEYS}
    )
    body = r.json()
    assert r.status_code == 200
    assert body["trigger_frame_index"] == 3
    assert body["details"]["decision"]["trigger_tube_id"] == 7
    assert body["details"]["tubes"][0]["first_crossing_frame"] == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd api && uv run pytest tests/test_app.py -q -k compute_trigger`
Expected: FAILED — `trigger_frame_index` missing from the response and `runner.compute_trigger` stays `False`/unset (the route does not accept the param yet; FastAPI ignores unknown query params).

- [ ] **Step 3: Implement the route param**

In `api/src/temporal_model/api/app.py`, change the route signature (line 129):

```python
async def predict(
    body: PredictRequest,
    request: Request,
    verbose: bool = False,
    compute_trigger: bool = False,
) -> PredictResponse:
```

Forward it to the runner (line 156):

```python
            out = await runner.predict(
                paths,
                roi=body.roi_xyxyn,
                timer=timer,
                profile=profile,
                compute_trigger=compute_trigger,
            )
```

and to the mapper (line 170):

```python
            return to_response(
                out,
                api_version=settings.api_version,
                model_version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                compute_trigger=compute_trigger,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                profiling=profiling,
            )
```

- [ ] **Step 4: Run the full api suite**

Run: `cd api && uv run pytest -q`
Expected: all pass, including the untouched `test_predict_default` / `test_predict_verbose` exact-shape asserts.

- [ ] **Step 5: Commit**

```bash
git add api/tests/test_app.py api/src/temporal_model/api/app.py
git commit -m "feat(api): add compute_trigger query param to /predict (#26)"
```

---

### Task 4: Document the flag and run the final checks

**Files:**
- Modify: `api/README.md:8-22`

- [ ] **Step 1: Update the endpoint docs**

In `api/README.md`, extend the `/predict` bullet — after the `?verbose=true` sentence (line 20-22), change:

```markdown
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). See `docs/specs/2026-06-02-api-service-design.md` for the
  full contract.
```

to:

```markdown
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). `POST /predict?compute_trigger=true` runs the
  first-crossing search (extra classifier work, off by default) and adds a
  top-level `trigger_frame_index` (`null` if nothing crossed) — with
  `verbose=true` it also fills `details.decision.trigger_tube_id` and
  per-tube `details.tubes[].first_crossing_frame`. See
  `docs/specs/2026-06-02-api-service-design.md` for the full contract.
```

- [ ] **Step 2: Lint and full test suite**

Run: `cd api && make lint && make test`
Expected: ruff clean, all tests pass.

- [ ] **Step 3: Commit**

```bash
git add api/README.md
git commit -m "docs(api): document the compute_trigger flag"
```
