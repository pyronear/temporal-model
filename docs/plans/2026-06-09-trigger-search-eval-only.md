# Trigger search is eval-only (`compute_trigger`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate the `trigger_search` stage behind a `compute_trigger` flag (default `False`) so production `predict()` skips the ~5 s / 34% prefix re-scoring loop; eval opts in to keep the time-to-detection metric.

**Architecture:** `predict()` already has every per-tube full-length logit/prob it needs to decide `is_positive`; the prefix search only adds time-to-detection data. We extract the per-tube decision predicate into a shared helper, then branch in `predict()`: skip the search and report `trigger_*` as `None` when `compute_trigger=False`, run the unchanged `find_first_crossing_trigger` when `True`. The top-level API `probability` drops its dependency on `trigger_tube_id` and becomes `max(kept-tube prob)` unconditionally.

**Tech Stack:** Python 3.11, PyTorch, pytest. Packages: `core`, `api`, `eval`.

**Spec:** `docs/specs/2026-06-09-trigger-search-eval-only-design.md`

---

## File Structure

- `core/src/temporal_model/core/inference.py` — add `make_decision_fn`; refactor `find_first_crossing_trigger` to use it (no behavior change).
- `core/src/temporal_model/core/model.py` — add `compute_trigger: bool = False` to `predict()`; branch the `trigger_search` stage.
- `api/src/temporal_model/api/schemas.py` — simplify `_decision_probability` to max-everywhere; drop its `is_smoke` param; update the one caller.
- `eval/src/temporal_model/eval/evaluate.py` — pass `compute_trigger=True`.
- Tests: `core/tests/test_inference_units.py`, `core/tests/test_model_edge_cases.py`, `api/tests/test_schemas.py`, `eval/tests/test_evaluate_driver.py`.

**Unaffected (verified, no change needed):** `core/tests/test_model_parity.py` (asserts kept-tube logits, not trigger), `core/tests/test_details_schema.py` (constructs schema objects directly), `eval/tests/test_protocol_eval.py` (constructs `TemporalModelOutput` directly), `train/package_predict.py` and `api/model_runner.py` (inherit the `False` default; only read tube structure / pass through).

---

## Task 1: Extract shared decision predicate `make_decision_fn`

**Files:**
- Modify: `core/src/temporal_model/core/inference.py`
- Test: `core/tests/test_inference_units.py`

- [ ] **Step 1: Write the failing test**

Add to `core/tests/test_inference_units.py` (top-level, near the other inference unit tests). The import on line 16 currently pulls `find_first_crossing_trigger`; extend it to also import `make_decision_fn`:

```python
from temporal_model.core.inference import (  # adjust existing import block
    find_first_crossing_trigger,
    make_decision_fn,
)
```

```python
def test_make_decision_fn_max_logit():
    decides = make_decision_fn(
        "max_logit", threshold=0.5, calibrator=None, logistic_threshold=0.5
    )
    assert decides(0.6, None, 1) is True
    assert decides(0.4, None, 1) is False


def test_make_decision_fn_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="unknown aggregation"):
        make_decision_fn(
            "bogus", threshold=0.0, calibrator=None, logistic_threshold=0.5
        )


def test_make_decision_fn_logistic_requires_calibrator():
    with pytest.raises(ValueError, match="requires a fitted calibrator"):
        make_decision_fn(
            "logistic", threshold=0.0, calibrator=None, logistic_threshold=0.5
        )
```

(`pytest` is already imported in this file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest core/tests/test_inference_units.py::test_make_decision_fn_max_logit -v`
Expected: FAIL with `ImportError: cannot import name 'make_decision_fn'`.

- [ ] **Step 3: Add `make_decision_fn` and refactor `find_first_crossing_trigger`**

In `core/src/temporal_model/core/inference.py`, add the `Callable` import. The top of the file has `from typing import Any` — add above it:

```python
from collections.abc import Callable
```

Add this function immediately **before** `def find_first_crossing_trigger(` (around line 304):

```python
def make_decision_fn(
    aggregation: str,
    *,
    threshold: float,
    calibrator: LogisticCalibrator | None,
    logistic_threshold: float,
) -> Callable[[float, Tube, int], bool]:
    """Build the per-tube positive-decision predicate for an aggregation rule.

    Returns ``decides_positive(logit, tube, n_tubes) -> bool``. Shared by the
    full-tube decision (production fast path) and the prefix re-scoring loop
    (:func:`find_first_crossing_trigger`), so both agree on the rule.

    Raises:
        ValueError: unknown ``aggregation`` or ``"logistic"`` without a
            calibrator.
    """
    if aggregation == "max_logit":

        def decides_positive(logit: float, _tube: Tube, _n_tubes: int) -> bool:
            return logit >= threshold

        return decides_positive
    if aggregation == "logistic":
        if calibrator is None:
            raise ValueError("aggregation='logistic' requires a fitted calibrator")

        def decides_positive(logit: float, tube: Tube, n_tubes: int) -> bool:
            features = extract_features(
                tube_feature_dict(tube, logit), n_tubes=n_tubes
            )
            return bool(calibrator.predict_proba(features) >= logistic_threshold)

        return decides_positive
    raise ValueError(f"unknown aggregation: {aggregation!r}")
```

Then inside `find_first_crossing_trigger`, replace the inline predicate block (currently the `if aggregation == "max_logit": ... else: raise ValueError(...)` spanning roughly lines 361–375, ending just before `n_tubes = len(tubes)`):

```python
    if aggregation == "max_logit":

        def decides_positive(logit: float, _tube_prefix: Tube, _n_tubes: int) -> bool:
            return logit >= threshold
    elif aggregation == "logistic":
        if calibrator is None:
            raise ValueError("aggregation='logistic' requires a fitted calibrator")

        def decides_positive(logit: float, tube_prefix: Tube, n_tubes: int) -> bool:
            features = extract_features(
                tube_feature_dict(tube_prefix, logit), n_tubes=n_tubes
            )
            return bool(calibrator.predict_proba(features) >= logistic_threshold)
    else:
        raise ValueError(f"unknown aggregation: {aggregation!r}")
```

with:

```python
    decides_positive = make_decision_fn(
        aggregation,
        threshold=threshold,
        calibrator=calibrator,
        logistic_threshold=logistic_threshold,
    )
```

Leave the `if not tubes: return False, None, None, {}` guard above it untouched, so the early-return-before-validation behavior is preserved.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest core/tests/test_inference_units.py -v`
Expected: PASS — the new `make_decision_fn` tests AND all existing `find_first_crossing_trigger` tests (the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add core/src/temporal_model/core/inference.py core/tests/test_inference_units.py
git commit -m "refactor(core): extract make_decision_fn shared by trigger search"
```

---

## Task 2: Gate the trigger search behind `compute_trigger` in `predict()`

**Files:**
- Modify: `core/src/temporal_model/core/model.py`
- Test: `core/tests/test_model_edge_cases.py`

- [ ] **Step 1: Write the failing test**

Add to `core/tests/test_model_edge_cases.py` inside the same test class as `test_predict_details_include_per_tube_entries` (it has the `tiny_classifier` / `red_frames` fixtures and the `threshold: -1e6` positive-forcing pattern):

```python
    def test_compute_trigger_gate(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        """Default predict() skips the trigger search (trigger fields None);
        compute_trigger=True restores the full trigger output, with the same
        is_positive in both modes."""
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(per_frame)
        cfg = {
            **TEST_CONFIG,
            "decision": {**TEST_CONFIG["decision"], "threshold": -1e6},
        }
        model = BboxTubeTemporalModel(
            yolo_model=yolo,
            classifier=tiny_classifier,
            config=cfg,
            device="cpu",
        )

        default = model.predict(frames=red_frames)
        assert default.is_positive is True
        assert default.trigger_frame_index is None
        assert default.details["decision"]["trigger_tube_id"] is None
        assert default.details["tubes"]["kept"][0]["first_crossing_frame"] is None

        full = model.predict(frames=red_frames, compute_trigger=True)
        assert full.is_positive is default.is_positive
        assert full.trigger_frame_index is not None
        assert full.details["decision"]["trigger_tube_id"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest core/tests/test_model_edge_cases.py -k compute_trigger_gate -v`
Expected: FAIL — `predict()` does not accept `compute_trigger` (TypeError), or returns a non-None trigger by default.

- [ ] **Step 3: Implement the gate**

In `core/src/temporal_model/core/model.py`, extend the import on line 23 to add `make_decision_fn`:

```python
from .inference import (  # existing block already imports find_first_crossing_trigger
    ...,
    find_first_crossing_trigger,
    make_decision_fn,
    ...,
)
```

Change the `predict` signature (line 144) from:

```python
    def predict(
        self, frames: list[Frame], *, timer: StageTimer | None = None
    ) -> TemporalModelOutput:
```

to:

```python
    def predict(
        self,
        frames: list[Frame],
        *,
        timer: StageTimer | None = None,
        compute_trigger: bool = False,
    ) -> TemporalModelOutput:
```

Replace the `with stage_ctx(timer, "trigger_search"):` block (currently lines 285–301) with:

```python
        with stage_ctx(timer, "trigger_search"):
            if compute_trigger:
                is_positive, trigger, trigger_tube_id, per_tube_first_crossing = (
                    find_first_crossing_trigger(
                        classifier=self._classifier,
                        tubes=kept,
                        patches_per_tube=patches_per_tube,
                        masks_per_tube=masks_per_tube,
                        full_logits=logits,
                        aggregation=aggregation,
                        threshold=float(dec["threshold"]),
                        calibrator=self._calibrator,
                        logistic_threshold=float(
                            dec.get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)
                        ),
                        min_prefix_length=tubes_cfg["infer_min_tube_length"],
                    )
                )
            else:
                decides_positive = make_decision_fn(
                    aggregation,
                    threshold=float(dec["threshold"]),
                    calibrator=self._calibrator,
                    logistic_threshold=float(
                        dec.get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)
                    ),
                )
                n_kept = len(kept)
                is_positive = any(
                    decides_positive(float(logits[i].item()), tube, n_kept)
                    for i, tube in enumerate(kept)
                )
                trigger = None
                trigger_tube_id = None
                per_tube_first_crossing = {}
```

The downstream code is unchanged: `per_tube_first_crossing.get(tube.tube_id, {}).get("crossing_frame")` yields `None` for every tube when the dict is empty, so `first_crossing_frame=None`; `trigger_tube_id=None` flows into `_make_details`; `trigger=None` becomes `trigger_frame_index`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest core/tests/test_model_edge_cases.py -k compute_trigger_gate -v`
Expected: PASS.

- [ ] **Step 5: Fix the two existing trigger-semantics tests to opt in**

These assert trigger output via the default path and now need `compute_trigger=True`.

In `test_predict_details_include_per_tube_entries` (~line 333), change:

```python
        out = model.predict(frames=red_frames)
```
to:
```python
        out = model.predict(frames=red_frames, compute_trigger=True)
```

In `test_first_crossing_trigger_never_exceeds_end_frame` (~line 459), change:

```python
        out = model.predict(frames=red_frames)
```
to:
```python
        out = model.predict(frames=red_frames, compute_trigger=True)
```

(The negative-case tests at lines ~139, ~156 assert `trigger_frame_index is None` / `is_positive is False`, which the default path already satisfies — leave them.)

- [ ] **Step 6: Run the full core edge-case + parity suite**

Run: `uv run pytest core/tests/test_model_edge_cases.py core/tests/test_model_parity.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add core/src/temporal_model/core/model.py core/tests/test_model_edge_cases.py
git commit -m "feat(core): gate trigger search behind compute_trigger (default off)"
```

---

## Task 3: `probability` = max kept-tube prob unconditionally

**Files:**
- Modify: `api/src/temporal_model/api/schemas.py`
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Update the tests to the new contract**

In `api/tests/test_schemas.py`, rewrite `test_smoke_default_uses_trigger_tube_probability` (line 55) so the trigger tube is **not** the max-prob tube, locking the max-everywhere semantics. Replace the whole test with:

```python
def test_smoke_uses_max_kept_probability():
    # Trigger tube (id 7) has the LOWER prob; reported value is the max (0.91).
    out = SimpleNamespace(
        is_positive=True,
        trigger_frame_index=3,
        details=_details([_tube(7, 0.62), _tube(2, 0.91)]),
    )
    resp = to_response(out, name="m", version="1", calibrated=True, verbose=False)
    body = resp.model_dump()
    assert body["is_smoke"] is True
    assert body["probability"] == 0.91
    assert body["trigger_frame_index"] == 3
```

Delete `test_smoke_trigger_tube_missing_returns_none` (lines ~97–102) entirely — its scenario (probability is `None` when the trigger tube is absent from kept) no longer exists under max-everywhere semantics.

Leave `test_negative_uses_max_kept_probability`, `test_negative_no_tubes_is_zero_when_calibrated`, and `test_uncalibrated_probability_is_null` as-is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest api/tests/test_schemas.py::test_smoke_uses_max_kept_probability -v`
Expected: FAIL — current code returns the trigger tube's prob (0.62), not 0.91.

- [ ] **Step 3: Simplify `_decision_probability` and its caller**

In `api/src/temporal_model/api/schemas.py`, replace the whole `_decision_probability` function (lines 85–104) with:

```python
def _decision_probability(
    details: dict[str, Any], calibrated: bool
) -> float | None:
    """Top-level probability per the API contract.

    None if uncalibrated. Otherwise the max kept-tube probability (0.0 when no
    tubes were kept), regardless of the smoke decision.
    """
    if not calibrated:
        return None
    kept = details["tubes"]["kept"]
    probs = [t["probability"] for t in kept if t.get("probability") is not None]
    return max(probs) if probs else 0.0
```

Update the caller in `to_response` (line 144) from:

```python
        "probability": _decision_probability(out.details, out.is_positive, calibrated),
```
to:
```python
        "probability": _decision_probability(out.details, calibrated),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest api/tests/test_schemas.py api/tests/test_app.py -v`
Expected: PASS (the single-kept-tube fixtures in `test_app.py` have max == trigger-tube prob, so they are unaffected).

- [ ] **Step 5: Commit**

```bash
git add api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): report max kept-tube probability, drop trigger_tube_id dependency"
```

---

## Task 4: Eval opts into `compute_trigger=True`

**Files:**
- Modify: `eval/src/temporal_model/eval/evaluate.py:102`
- Test: `eval/tests/test_evaluate_driver.py`

- [ ] **Step 1: Write the failing test**

In `eval/tests/test_evaluate_driver.py`, first update `_FakeModel.predict` (line 47) to accept the new keyword and record it:

```python
    def predict(self, frames, *, compute_trigger=False):
        self.last_compute_trigger = compute_trigger
        ...  # body unchanged
```

Then add a test (it can reuse the existing fixtures/helpers the other driver tests use to build `sequences_dir` and call the entrypoint — mirror the setup of the test around line 120):

```python
def test_evaluate_requests_trigger(tmp_path, monkeypatch):
    # Mirror the setup used by the existing driver test, then assert the model
    # was asked for the trigger output (TTD needs trigger_frame_index).
    model = _FakeModel()
    _run_evaluate_with(model, tmp_path, monkeypatch)  # existing harness helper
    assert model.last_compute_trigger is True
```

If the existing driver test does not expose a reusable `_run_evaluate_with` helper, instead assert inside the existing end-to-end test (around line 120) after the run completes:

```python
    assert model.last_compute_trigger is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest eval/tests/test_evaluate_driver.py -k trigger -v`
Expected: FAIL — `last_compute_trigger` is `False` (evaluate still calls `predict(frames)`).

- [ ] **Step 3: Pass the flag in the driver**

In `eval/src/temporal_model/eval/evaluate.py`, change line 102 from:

```python
        output = model.predict(frames)
```
to:
```python
        output = model.predict(frames, compute_trigger=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest eval/tests/test_evaluate_driver.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/src/temporal_model/eval/evaluate.py eval/tests/test_evaluate_driver.py
git commit -m "feat(eval): request compute_trigger=True so TTD metric is preserved"
```

---

## Task 5: Final verification

- [ ] **Step 1: Run the full test suite across the touched packages**

Run: `uv run pytest core api eval -q`
Expected: all PASS, no skips beyond the pre-existing CUDA skip.

- [ ] **Step 2: Lint the changed files**

Run: `uv run ruff check core/src/temporal_model/core/inference.py core/src/temporal_model/core/model.py api/src/temporal_model/api/schemas.py eval/src/temporal_model/eval/evaluate.py`
Expected: no errors (in particular, no unused-argument warning on `_decision_probability`).

- [ ] **Step 3: Confirm the prod fast path drops the stage (manual sanity)**

Run: `uv run temporal-benchmark core --help` to confirm the CLI still loads, then (if a fixture sequence store is available locally) run a small `temporal-benchmark core` and confirm the `trigger_search` stage share is ~0 with the default path. If no store is available, note this as deferred to CI/VM.

Expected: `trigger_search` ≈ 0% of latency on the default path.

---

## Self-Review

- **Spec coverage:** §1 `compute_trigger` → Task 2; §2 shared predicate → Task 1; §3 probability contract → Task 3; §4 call sites (eval) → Task 4, (api/model_runner, package_predict, benchmark) → inherit default, no task needed; §5 stage timer kept in both branches → Task 2 Step 3; testing list → Tasks 1–4 + Task 5. Out-of-scope items (batching, protocol threading) intentionally have no task.
- **Placeholder scan:** none — all steps carry real code/commands. Task 4 Step 1 offers two concrete wiring options because the existing test harness shape determines which applies; both are spelled out.
- **Type/name consistency:** `make_decision_fn(aggregation, *, threshold, calibrator, logistic_threshold) -> Callable[[float, Tube, int], bool]` used identically in Task 1 (definition + `find_first_crossing_trigger`) and Task 2 (`predict`). `_decision_probability(details, calibrated)` signature matches its updated caller. `compute_trigger` keyword spelled identically across `predict`, the gate test, the edge-case opt-ins, `_FakeModel.predict`, and `evaluate.py`.
