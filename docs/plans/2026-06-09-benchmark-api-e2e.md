# Benchmark Phase 2 — API e2e Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure realistic production serving latency (HTTP → S3 fetch → cached detection → classifier → serialization) on the CPU VM, with a server-side per-stage breakdown under cold and warm cache regimes.

**Architecture:** Add an opt-in `TEMPORAL_API_PROFILE` flag to the API that threads a request-scoped `StageTimer` (reusing `core.stage_timer`) through `app → ModelRunner → core.predict`, timing `s3_fetch / detector / pad / tubes / crop / classifier` plus cache hit/miss counts, surfaced in logs and the `?verbose=true` `details.profiling` field. A new `benchmark/run_api.py` drives the API over HTTP in cold (full-sequence) and warm (growing-prefix) passes; `report.py` aggregates the two passes separately.

**Tech Stack:** FastAPI, pydantic, boto3/MinIO, requests (new benchmark dep), pandas, Docker Compose.

**Spec:** `docs/specs/2026-06-09-benchmark-api-e2e-design.md`.

---

## File Structure

**API (edited):**
- `api/src/temporal_model/api/settings.py` — add `profile: bool = False`.
- `api/src/temporal_model/api/schemas.py` — optional `profiling` field on `Details`; `to_response(profiling=...)`.
- `api/src/temporal_model/api/model_runner.py` — `predict(..., timer=None, profile=None)`; time `detector`, record cache counts, thread `timer` into `model.predict`.
- `api/src/temporal_model/api/app.py` — build request-scoped `StageTimer`, time `s3_fetch`, assemble + log profiling.

**Benchmark (new/edited):**
- `benchmark/pyproject.toml` — add `requests`.
- `benchmark/src/temporal_model/benchmark/run_api.py` — cold/warm HTTP client (new).
- `benchmark/src/temporal_model/benchmark/report.py` — add `summarize_api` / `write_api_report`.
- `benchmark/src/temporal_model/benchmark/cli.py` — add `api` subcommand.
- `benchmark/scripts/provision_api_vm.sh`, `benchmark/scripts/upload_frames_to_minio.py` (new).
- `benchmark/tests/test_run_api.py`, `benchmark/tests/test_report_api.py` (new).
- `api/tests/test_app.py` — profiling on/off tests.

**Profiling payload shape** (returned in `details.profiling`, logged as JSON):
```json
{
  "stages_ms": {"s3_fetch": 0.0, "detector": 0.0, "pad": 0.0, "tubes": 0.0, "crop": 0.0, "classifier": 0.0},
  "total_ms": 0.0,
  "n_frames": 0, "cache_hits": 0, "cache_misses": 0
}
```

---

## Task 1: API profiling setting + response schema field

**Files:**
- Modify: `api/src/temporal_model/api/settings.py`
- Modify: `api/src/temporal_model/api/schemas.py`
- Test: `api/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_schemas.py`:

```python
def test_to_response_includes_profiling_when_verbose():
    from types import SimpleNamespace

    from temporal_model.api.schemas import to_response

    details = {
        "decision": {"aggregation": "max_logit", "threshold": 0.5, "trigger_tube_id": None},
        "preprocessing": {"num_frames_input": 6, "num_truncated": 0, "padded_frame_indices": []},
        "tubes": {"num_candidates": 0, "kept": []},
    }
    out = SimpleNamespace(is_positive=False, trigger_frame_index=None, details=details)
    profiling = {"stages_ms": {"s3_fetch": 1.0}, "total_ms": 1.0, "n_frames": 6,
                 "cache_hits": 4, "cache_misses": 2}

    resp = to_response(out, name="m", version="1", calibrated=False, verbose=True,
                       profiling=profiling)
    assert resp.details.profiling == profiling

    # Omitted when not verbose, and harmless when profiling is None.
    resp2 = to_response(out, name="m", version="1", calibrated=False, verbose=False,
                        profiling=profiling)
    assert resp2.details is None
    resp3 = to_response(out, name="m", version="1", calibrated=False, verbose=True,
                        profiling=None)
    assert resp3.details.profiling is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && uv run pytest tests/test_schemas.py::test_to_response_includes_profiling_when_verbose -v`
Expected: FAIL — `to_response() got an unexpected keyword argument 'profiling'`

- [ ] **Step 3: Add `profile` to settings**

In `api/src/temporal_model/api/settings.py`, after the `detection_cache_size` field:

```python
    # Per-frame detection LRU capacity (frame_id → detections). 0 disables.
    detection_cache_size: int = 4096

    # When true, record per-stage timing + cache counts for each request and
    # surface them in logs and the verbose response (`details.profiling`).
    profile: bool = False
```

- [ ] **Step 4: Add the `profiling` field + plumb it through `to_response`**

In `api/src/temporal_model/api/schemas.py`, add the field to `Details`:

```python
class Details(BaseModel):
    decision: Decision
    preprocessing: Preprocessing
    tubes: list[Tube]
    profiling: dict[str, Any] | None = None
```

Update `_to_details` to accept and pass it (add the parameter and the field):

```python
def _to_details(
    details: dict[str, Any],
    *,
    threshold_overridden: bool,
    packaged_threshold: float | None,
    profiling: dict[str, Any] | None = None,
) -> Details:
    tubes_block = details["tubes"]
    pre = details["preprocessing"]
    return Details(
        decision=Decision(
            **details["decision"],
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
        ),
        preprocessing=Preprocessing(
            num_frames_input=pre["num_frames_input"],
            num_truncated=pre["num_truncated"],
            padded_frame_indices=pre["padded_frame_indices"],
            num_tube_candidates=tubes_block["num_candidates"],
        ),
        tubes=[Tube(**t) for t in tubes_block["kept"]],
        profiling=profiling,
    )
```

Add the `profiling` parameter to `to_response` and forward it:

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
    profiling: dict[str, Any] | None = None,
) -> PredictResponse:
    """Reshape a core model output into the public response DTO."""
    kwargs: dict[str, Any] = {
        "is_smoke": out.is_positive,
        "probability": _decision_probability(out.details, calibrated),
        "model": ModelInfo(name=name, version=version),
    }
    if verbose:
        kwargs["details"] = _to_details(
            out.details,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
            profiling=profiling,
        )
    return PredictResponse(**kwargs)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd api && uv run pytest tests/test_schemas.py -q`
Expected: PASS

- [ ] **Step 6: Lint + commit**

```bash
cd api && uv run ruff check src/temporal_model/api/settings.py src/temporal_model/api/schemas.py tests/test_schemas.py && uv run ruff format src/temporal_model/api/settings.py src/temporal_model/api/schemas.py tests/test_schemas.py
cd .. && git add api/src/temporal_model/api/settings.py api/src/temporal_model/api/schemas.py api/tests/test_schemas.py
git commit -m "feat(api): add TEMPORAL_API_PROFILE setting + optional details.profiling field"
```

---

## Task 2: ModelRunner — time detection + record cache counts

Thread an optional `timer` + `profile` dict through `ModelRunner.predict` / `_predict_sync`: time the real `detect()` call under `"detector"`, write cache counts into `profile`, and pass `timer` into `model.predict`.

**Files:**
- Modify: `api/src/temporal_model/api/model_runner.py`
- Test: `api/tests/test_model_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `api/tests/test_model_runner.py`:

```python
import asyncio

from temporal_model.core.stage_timer import StageTimer


class _StubFrame:
    def __init__(self, fid):
        self.frame_id = fid


class _StubModel:
    """Minimal model exposing the load_sequence/detect/predict surface."""

    def __init__(self):
        self.predict_timer = None

    def load_sequence(self, paths):
        return [_StubFrame(str(p)) for p in paths]

    def detect(self, misses):
        # one FrameDetections-like object per miss, with a frame_id attr
        from types import SimpleNamespace
        return [SimpleNamespace(frame_id=f.frame_id) for f in misses]

    def predict(self, frames, *, frame_detections=None, timer=None):
        self.predict_timer = timer
        if timer is not None:
            with timer.stage("classifier"):
                pass
        from types import SimpleNamespace
        return SimpleNamespace(is_positive=False, trigger_frame_index=None, details={})


def _runner_with(model, cache_size=4096):
    return ModelRunner(model, name="m", version="1", calibrated=False,
                       detection_cache_size=cache_size)


def test_predict_records_detector_timing_and_cache_counts():
    model = _StubModel()
    runner = _runner_with(model)
    timer = StageTimer()
    profile: dict = {}

    asyncio.run(runner.predict(["a", "b", "c"], timer=timer, profile=profile))

    timings = timer.as_dict()
    assert "detector" in timings           # detect() was timed
    assert model.predict_timer is timer    # same timer threaded into predict()
    assert profile["n_frames"] == 3
    assert profile["cache_misses"] == 3    # cold: all miss
    assert profile["cache_hits"] == 0


def test_second_predict_hits_cache():
    model = _StubModel()
    runner = _runner_with(model)
    asyncio.run(runner.predict(["a", "b"]))         # warms the cache
    profile: dict = {}
    asyncio.run(runner.predict(["a", "b"], profile=profile))
    assert profile["cache_hits"] == 2
    assert profile["cache_misses"] == 0


def test_predict_without_timer_still_works():
    model = _StubModel()
    runner = _runner_with(model)
    out = asyncio.run(runner.predict(["a"]))
    assert out.is_positive is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && uv run pytest tests/test_model_runner.py::test_predict_records_detector_timing_and_cache_counts -v`
Expected: FAIL — `predict() got an unexpected keyword argument 'timer'`

- [ ] **Step 3: Implement the threading**

In `api/src/temporal_model/api/model_runner.py`, add the import near the top (after the existing imports):

```python
from temporal_model.core.stage_timer import StageTimer, stage_ctx
```

Replace `predict` and `_predict_sync` with:

```python
    async def predict(
        self,
        frame_paths: list[Path],
        *,
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        """Resolve detections (cache + detect misses) then run the model.

        The whole orchestration runs in a worker thread under the lock, so the
        cache is accessed by one prediction at a time. When ``timer``/``profile``
        are supplied, the ``detector`` stage is timed and cache counts recorded.
        """
        async with self._lock:
            return await run_in_threadpool(
                self._predict_sync, frame_paths, timer, profile
            )

    def _predict_sync(
        self,
        frame_paths: list[Path],
        timer: StageTimer | None = None,
        profile: dict[str, Any] | None = None,
    ) -> Any:
        started = time.perf_counter()
        frames = self._model.load_sequence(frame_paths)
        resolved: dict[str, Any] = {}
        misses = []
        for f in frames:
            if f.frame_id in self._cache:
                resolved[f.frame_id] = self._cache.get(f.frame_id)
            else:
                misses.append(f)
        with stage_ctx(timer, "detector"):
            detected = self._model.detect(misses)
        for fd in detected:
            self._cache.put(fd.frame_id, fd)
            resolved[fd.frame_id] = fd
        out = self._model.predict(frames, frame_detections=resolved, timer=timer)
        if profile is not None:
            profile["n_frames"] = len(frames)
            profile["cache_hits"] = len(frames) - len(misses)
            profile["cache_misses"] = len(misses)
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd api && uv run pytest tests/test_model_runner.py -q`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
cd api && uv run ruff check src/temporal_model/api/model_runner.py tests/test_model_runner.py && uv run ruff format src/temporal_model/api/model_runner.py tests/test_model_runner.py
cd .. && git add api/src/temporal_model/api/model_runner.py api/tests/test_model_runner.py
git commit -m "feat(api): time detection + record cache counts in ModelRunner when profiling"
```

---

## Task 3: app.py — assemble + surface profiling

Build the request-scoped `StageTimer`, time `s3_fetch`, assemble the profiling payload, log it, and pass it to `to_response`. Off by default → unchanged behavior.

**Files:**
- Modify: `api/src/temporal_model/api/app.py`
- Test: `api/tests/test_app.py`

- [ ] **Step 1: Write the failing test**

In `api/tests/test_app.py`, make `FakeRunner.predict` accept the new kwargs and populate them, then add the profiling tests. First update `FakeRunner.predict`:

```python
    async def predict(self, paths, *, timer=None, profile=None):
        if self._error:
            raise self._error
        if timer is not None:
            with timer.stage("detector"):
                pass
        if profile is not None:
            profile.update(n_frames=len(paths), cache_hits=0, cache_misses=len(paths))
        return self._output
```

Add at the top of the file:

```python
from temporal_model.core.stage_timer import StageTimer  # noqa: F401  (import check)
```

Append these tests:

```python
def test_predict_profiling_off_by_default(client):
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json()["details"].get("profiling") is None


def test_predict_profiling_on_surfaces_block(client, monkeypatch):
    monkeypatch.setattr(settings, "profile", True)
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    prof = r.json()["details"]["profiling"]
    assert "s3_fetch" in prof["stages_ms"]
    assert "detector" in prof["stages_ms"]
    assert prof["n_frames"] == len(KEYS)
    assert prof["cache_misses"] == len(KEYS)
    assert prof["total_ms"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd api && uv run pytest tests/test_app.py::test_predict_profiling_on_surfaces_block -v`
Expected: FAIL — `profiling` is `None` (app not wired yet)

- [ ] **Step 3: Wire app.py**

In `api/src/temporal_model/api/app.py`, add imports:

```python
import json

from temporal_model.core.stage_timer import StageTimer, stage_ctx
```

Replace the body of the `predict` handler's `try` block with:

```python
    with tempfile.TemporaryDirectory() as tmp:
        try:
            # StageTimer with no device → no GPU syncs (the served target is CPU);
            # exact wall-clock timing for s3_fetch + the CPU model stages.
            timer = StageTimer() if settings.profile else None
            profile: dict | None = {} if settings.profile else None

            with stage_ctx(timer, "s3_fetch"):
                # fetch_frames is blocking boto3 I/O — run it off the event loop.
                paths = await run_in_threadpool(
                    fetch_frames, s3_client, settings.s3_bucket, body.frames, Path(tmp)
                )

            out = await runner.predict(paths, timer=timer, profile=profile)

            profiling = None
            if timer is not None:
                stages = timer.as_dict()
                profiling = {
                    "stages_ms": stages,
                    "total_ms": round(sum(stages.values()), 3),
                    **(profile or {}),
                }
                logger.info("profile %s", json.dumps(profiling))

            return to_response(
                out,
                name=runner.name,
                version=runner.version,
                calibrated=runner.calibrated,
                verbose=verbose,
                threshold_overridden=runner.threshold_overridden,
                packaged_threshold=runner.packaged_threshold,
                profiling=profiling,
            )
        except ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as inference_error
            raise InferenceError(str(exc)) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/test_app.py -q`
Expected: PASS (existing tests still pass; the two new profiling tests pass)

- [ ] **Step 5: Run the full API suite (no regressions)**

Run: `cd api && uv run pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Lint + commit**

```bash
cd api && uv run ruff check src/temporal_model/api/app.py tests/test_app.py && uv run ruff format src/temporal_model/api/app.py tests/test_app.py
cd .. && git add api/src/temporal_model/api/app.py api/tests/test_app.py
git commit -m "feat(api): assemble + surface request profiling (s3_fetch + stages + cache) when enabled"
```

---

## Task 4: benchmark — `run_api.py` client (cold + warm)

**Files:**
- Modify: `benchmark/pyproject.toml`
- Create: `benchmark/src/temporal_model/benchmark/run_api.py`
- Test: `benchmark/tests/test_run_api.py`

- [ ] **Step 1: Add `requests` dependency**

In `benchmark/pyproject.toml`, add to `dependencies`:

```toml
    "requests>=2.31",
```

Run: `cd benchmark && uv sync`
Expected: installs `requests`, no errors.

- [ ] **Step 2: Write the failing test** (covers the pure request-builder + row assembly with an injected poster)

```python
# benchmark/tests/test_run_api.py
"""Tests for the API benchmark client's request planning + row assembly."""

from pathlib import Path

from temporal_model.benchmark.run_api import build_requests, frame_key, rows_for_sequence
from temporal_model.core.protocol import Frame


def _seq(store, n):
    frames = [
        Frame(frame_id=f"f{i}", image_path=store / "org/cam/seq" / f"{i}.jpg")
        for i in range(n)
    ]
    return frames


def test_frame_key_is_store_relative_posix(tmp_path):
    f = Frame(frame_id="x", image_path=tmp_path / "a/b/c.jpg")
    assert frame_key(tmp_path, f) == "a/b/c.jpg"


def test_build_requests_cold_is_single_full_list(tmp_path):
    frames = _seq(tmp_path, 5)
    reqs = build_requests(tmp_path, frames, "cold", warm_min_frames=3)
    assert len(reqs) == 1
    prefix_len, keys = reqs[0]
    assert prefix_len == 5
    assert len(keys) == 5


def test_build_requests_warm_is_growing_prefixes(tmp_path):
    frames = _seq(tmp_path, 5)
    reqs = build_requests(tmp_path, frames, "warm", warm_min_frames=3)
    assert [p for p, _ in reqs] == [3, 4, 5]
    assert [len(k) for _, k in reqs] == [3, 4, 5]


def test_build_requests_warm_short_sequence(tmp_path):
    frames = _seq(tmp_path, 2)  # shorter than warm_min_frames
    reqs = build_requests(tmp_path, frames, "warm", warm_min_frames=3)
    assert [p for p, _ in reqs] == [2]


def test_rows_for_sequence_flattens_profiling(tmp_path):
    frames = _seq(tmp_path, 3)

    def fake_post(url, keys):
        body = {
            "details": {
                "profiling": {
                    "stages_ms": {"s3_fetch": 5.0, "detector": 50.0, "classifier": 10.0},
                    "total_ms": 65.0,
                    "n_frames": len(keys),
                    "cache_hits": 0,
                    "cache_misses": len(keys),
                }
            }
        }
        return 200, body, 70.0  # status, json, e2e_ms

    rows = rows_for_sequence(
        tmp_path, "k1", frames, "cold", warm_min_frames=3,
        base_url="http://x", post=fake_post,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["pass"] == "cold"
    assert row["key"] == "k1"
    assert row["prefix_len"] == 3
    assert row["http_status"] == 200
    assert row["e2e_ms"] == 70.0
    assert row["detector_ms"] == 50.0
    assert row["total_ms"] == 65.0
    assert row["cache_misses"] == 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_run_api.py -q`
Expected: FAIL — `ModuleNotFoundError: ...benchmark.run_api`

- [ ] **Step 4: Implement `run_api.py`**

```python
# benchmark/src/temporal_model/benchmark/run_api.py
"""Drive the API over HTTP in cold and warm cache passes, collecting per-request
e2e latency and the server-side profiling block."""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from temporal_model.core.protocol import Frame

from .dataset import iter_sequences

logger = logging.getLogger(__name__)


def frame_key(store_dir: Path, frame: Frame) -> str:
    """S3 key for a frame: its path relative to the store root (POSIX)."""
    return frame.image_path.relative_to(store_dir).as_posix()


def build_requests(
    store_dir: Path,
    frames: list[Frame],
    pass_name: str,
    *,
    warm_min_frames: int,
) -> list[tuple[int, list[str]]]:
    """Plan the requests for one sequence under a cache pass.

    cold → one request with the full key list. warm → growing prefixes
    ``warm_min_frames .. N`` (clamped to N for short sequences).
    """
    keys = [frame_key(store_dir, f) for f in frames]
    n = len(keys)
    if pass_name == "cold":
        return [(n, keys)]
    start = min(warm_min_frames, n)
    return [(k, keys[:k]) for k in range(start, n + 1)]


def _http_post(base_url: str, keys: list[str]) -> tuple[int, dict, float]:
    """POST one request; return (status, json_body, e2e_ms)."""
    started = time.perf_counter()
    resp = requests.post(
        f"{base_url}/predict?verbose=true", json={"frames": keys}, timeout=300
    )
    e2e_ms = (time.perf_counter() - started) * 1000.0
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body, e2e_ms


def rows_for_sequence(
    store_dir: Path,
    key: str,
    frames: list[Frame],
    pass_name: str,
    *,
    warm_min_frames: int,
    base_url: str,
    post=_http_post,
) -> list[dict]:
    """Issue all requests for one sequence and return one raw row per request."""
    rows: list[dict] = []
    for prefix_len, keys in build_requests(
        store_dir, frames, pass_name, warm_min_frames=warm_min_frames
    ):
        status, body, e2e_ms = post(base_url, keys)
        row = {
            "pass": pass_name,
            "key": key,
            "prefix_len": prefix_len,
            "e2e_ms": round(e2e_ms, 3),
            "http_status": status,
        }
        prof = (body.get("details") or {}).get("profiling") if body else None
        if prof:
            for stage, ms in prof.get("stages_ms", {}).items():
                row[f"{stage}_ms"] = ms
            row["total_ms"] = prof.get("total_ms")
            row["n_frames"] = prof.get("n_frames")
            row["cache_hits"] = prof.get("cache_hits")
            row["cache_misses"] = prof.get("cache_misses")
        rows.append(row)
    return rows


def run_api(
    store_dir: Path,
    base_url: str,
    *,
    passes: tuple[str, ...] = ("cold", "warm"),
    warmup: int = 3,
    limit: int | None = None,
    warm_min_frames: int = 3,
) -> pd.DataFrame:
    """Benchmark the API over HTTP; one row per request, per pass."""
    sequences = list(iter_sequences(store_dir))
    if limit is not None:
        sequences = sequences[:limit]
    if not sequences:
        raise SystemExit(f"no sequences found under {store_dir}")

    for seq in sequences[:warmup]:
        rows_for_sequence(
            store_dir, seq.key, seq.frames, "cold",
            warm_min_frames=warm_min_frames, base_url=base_url,
        )

    rows: list[dict] = []
    for pass_name in passes:
        for i, seq in enumerate(sequences):
            try:
                rows.extend(
                    rows_for_sequence(
                        store_dir, seq.key, seq.frames, pass_name,
                        warm_min_frames=warm_min_frames, base_url=base_url,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — record + continue
                logger.warning("sequence %s (%s) failed: %s", seq.key, pass_name, exc)
                rows.append(
                    {"pass": pass_name, "key": seq.key, "http_status": 0, "e2e_ms": 0.0}
                )
            if (i + 1) % 25 == 0:
                logger.info("%s: %d/%d sequences", pass_name, i + 1, len(sequences))

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_run_api.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/run_api.py tests/test_run_api.py && uv run ruff format src/temporal_model/benchmark/run_api.py tests/test_run_api.py
cd .. && git add benchmark/pyproject.toml benchmark/uv.lock benchmark/src/temporal_model/benchmark/run_api.py benchmark/tests/test_run_api.py
git commit -m "feat(benchmark): add API e2e client with cold + warm cache passes"
```

---

## Task 5: benchmark — `summarize_api` + `write_api_report`

**Files:**
- Modify: `benchmark/src/temporal_model/benchmark/report.py`
- Test: `benchmark/tests/test_report_api.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_report_api.py
"""Tests for API benchmark aggregation."""

import json

import pandas as pd

from temporal_model.benchmark.report import summarize_api


def _row(pass_name, e2e, detector, status=200, hits=0, misses=6):
    return {
        "pass": pass_name, "key": "k", "prefix_len": 6, "e2e_ms": e2e,
        "http_status": status, "s3_fetch_ms": 5.0, "detector_ms": detector,
        "classifier_ms": 10.0, "total_ms": detector + 15.0,
        "n_frames": 6, "cache_hits": hits, "cache_misses": misses,
    }


def test_summarize_api_splits_cold_and_warm():
    df = pd.DataFrame([
        _row("cold", 100.0, 60.0, hits=0, misses=6),
        _row("cold", 200.0, 120.0, hits=0, misses=6),
        _row("warm", 30.0, 5.0, hits=5, misses=1),
        _row("warm", 50.0, 7.0, hits=6, misses=0),
    ])
    s = summarize_api(df)
    assert set(s["passes"]) == {"cold", "warm"}
    assert s["passes"]["cold"]["e2e_ms"]["p50"] == 200.0  # quantile(.5) of [100,200]
    assert s["passes"]["cold"]["n_requests"] == 2
    # warm amortizes the detector vs cold
    assert s["passes"]["warm"]["stage_ms_mean"]["detector"] < \
        s["passes"]["cold"]["stage_ms_mean"]["detector"]
    # warm cache hit rate = 11 hits / (11 hits + 1 miss)
    assert round(s["passes"]["warm"]["cache_hit_rate"], 3) == round(11 / 12, 3)
    # JSON-serializable
    assert json.loads(json.dumps(s)) == s


def test_summarize_api_counts_errors():
    df = pd.DataFrame([
        _row("cold", 100.0, 60.0),
        {"pass": "cold", "key": "k2", "http_status": 0, "e2e_ms": 0.0},
    ])
    s = summarize_api(df)
    assert s["passes"]["cold"]["n_errors"] == 1
    assert s["passes"]["cold"]["n_requests"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_report_api.py -q`
Expected: FAIL — `cannot import name 'summarize_api'`

- [ ] **Step 3: Implement `summarize_api` + `write_api_report`**

Append to `benchmark/src/temporal_model/benchmark/report.py`:

```python
def _api_stage_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_ms") and c not in ("e2e_ms", "total_ms")]


def _summarize_pass(pdf: pd.DataFrame) -> dict:
    ok = pdf[pdf["http_status"] == 200]
    e2e = ok["e2e_ms"] if not ok.empty else pd.Series(dtype=float)
    stage_cols = _api_stage_cols(pdf)
    stage_means = {
        c[:-3]: round(float(ok[c].mean()), 3) if (not ok.empty and c in ok) else 0.0
        for c in stage_cols
    }
    hits = float(ok["cache_hits"].sum()) if "cache_hits" in ok else 0.0
    misses = float(ok["cache_misses"].sum()) if "cache_misses" in ok else 0.0
    hit_rate = hits / (hits + misses) if (hits + misses) else 0.0
    mean_e2e = float(e2e.mean()) if not e2e.empty else 0.0
    return {
        "n_requests": int(len(pdf)),
        "n_errors": int((pdf["http_status"] != 200).sum()),
        "e2e_ms": {
            "p50": round(float(e2e.quantile(0.50)), 3) if not e2e.empty else 0.0,
            "p90": round(float(e2e.quantile(0.90)), 3) if not e2e.empty else 0.0,
            "p99": round(float(e2e.quantile(0.99)), 3) if not e2e.empty else 0.0,
            "mean": round(mean_e2e, 3),
        },
        "stage_ms_mean": stage_means,
        "cache_hit_rate": round(hit_rate, 4),
        "throughput_req_per_sec": round(1000.0 / mean_e2e, 3) if mean_e2e else 0.0,
    }


def summarize_api(df: pd.DataFrame) -> dict:
    """Aggregate API benchmark rows, split by cache pass (cold/warm)."""
    passes = {p: _summarize_pass(df[df["pass"] == p]) for p in df["pass"].unique()}
    return {"passes": passes, "n_requests": int(len(df))}


def write_api_report(
    df: pd.DataFrame,
    resources: pd.DataFrame,
    machine: dict,
    out_dir: Path,
) -> dict:
    """Write raw.parquet, resources.parquet, summary.json, report.md for the API run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "raw.parquet")
    resources.to_parquet(out_dir / "resources.parquet")
    summary = summarize_api(df)
    summary["machine"] = machine
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(_render_api_markdown(summary))
    return summary


def _render_api_markdown(summary: dict) -> str:
    m = summary["machine"]
    lines = [
        f"# API e2e benchmark — {m['hostname']}",
        "",
        f"- CPU: {m['cpu_model']} ({m['cpu_count_physical']} cores) · "
        f"device {m['device']} · {m['ram_total_gb']} GB",
        "",
    ]
    for name, p in summary["passes"].items():
        e = p["e2e_ms"]
        lines += [
            f"## {name} pass",
            f"- requests: {p['n_requests']} (errors: {p['n_errors']})",
            f"- e2e ms: p50 {e['p50']} · p90 {e['p90']} · p99 {e['p99']} · "
            f"mean {e['mean']}",
            f"- throughput: {p['throughput_req_per_sec']} req/s · "
            f"cache hit rate {p['cache_hit_rate']}",
            "- stage means (ms): "
            + " · ".join(f"{s} {v}" for s, v in p["stage_ms_mean"].items()),
            "",
        ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_report_api.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/report.py tests/test_report_api.py && uv run ruff format src/temporal_model/benchmark/report.py tests/test_report_api.py
cd .. && git add benchmark/src/temporal_model/benchmark/report.py benchmark/tests/test_report_api.py
git commit -m "feat(benchmark): aggregate API e2e results by cold/warm pass"
```

---

## Task 6: benchmark — `api` CLI subcommand

**Files:**
- Modify: `benchmark/src/temporal_model/benchmark/cli.py`

- [ ] **Step 1: Implement the subcommand**

In `benchmark/src/temporal_model/benchmark/cli.py`, add the import:

```python
from .report import write_api_report, write_report
from .run_api import run_api
from .run_core import resolve_device, run_core
```

(Replace the existing `from .report import write_report` and `from .run_core import ...` lines accordingly.)

Add the command handler:

```python
def _run_api_cmd(args: argparse.Namespace) -> None:
    with ResourceSampler(interval=args.sample_interval) as sampler:
        df = run_api(
            args.store,
            args.url,
            passes=tuple(args.passes.split(",")),
            warmup=args.warmup,
            limit=args.limit,
            warm_min_frames=args.warm_min_frames,
        )
    resources = pd.DataFrame(sampler.timeline())
    machine = machine_info(device="cpu")
    out_dir = args.out / f"{machine['hostname']}-api-{args.timestamp}"
    summary = write_api_report(df, resources, machine, out_dir)
    print(f"wrote {out_dir}")
    for name, p in summary["passes"].items():
        print(f"  {name}: p50 {p['e2e_ms']['p50']}ms · {p['n_errors']} errors")
```

In `main()`, after the `core` subparser block (before `args = ap.parse_args()`), add:

```python
    api = sub.add_parser("api", help="API end-to-end (HTTP) benchmark")
    api.add_argument("--url", default="http://localhost:8000")
    api.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    api.add_argument("--passes", default="cold,warm", help="comma list: cold,warm")
    api.add_argument("--warmup", type=int, default=3)
    api.add_argument("--limit", type=int, default=None)
    api.add_argument("--warm-min-frames", type=int, default=3)
    api.add_argument("--sample-interval", type=float, default=0.1)
    api.add_argument("--out", type=Path, default=Path("data/08_reporting"))
    api.add_argument("--timestamp", default="run")
    api.set_defaults(func=_run_api_cmd)
```

- [ ] **Step 2: Verify the CLI parses**

Run: `cd benchmark && uv run temporal-benchmark api --help`
Expected: prints the `api` subcommand help with `--url`, `--passes`, `--warm-min-frames`, etc.

- [ ] **Step 3: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/cli.py && uv run ruff format src/temporal_model/benchmark/cli.py
cd .. && git add benchmark/src/temporal_model/benchmark/cli.py
git commit -m "feat(benchmark): add 'api' CLI subcommand"
```

---

## Task 7: provisioning scripts (Docker + MinIO + frame upload)

**Files:**
- Create: `benchmark/scripts/provision_api_vm.sh`
- Create: `benchmark/scripts/upload_frames_to_minio.py`

- [ ] **Step 1: Create `provision_api_vm.sh`**

```bash
#!/usr/bin/env bash
# Stand up the API + MinIO compose stack with profiling on, on a VM.
# Usage: provision_api_vm.sh <ssh-host>
# Assumes the repo is already on the VM at ~/temporal-model with api/models/model.zip.
set -euo pipefail
HOST="${1:?usage: provision_api_vm.sh <ssh-host>}"

ssh "$HOST" 'command -v docker >/dev/null || (curl -fsSL https://get.docker.com | sudo sh)'
ssh "$HOST" 'sudo usermod -aG docker "$USER" || true'
# Bring the stack up with profiling enabled (detached).
ssh "$HOST" 'cd temporal-model/api && \
    TEMPORAL_API_PROFILE=true sudo -E docker compose up -d --build'
echo "stack up on $HOST — API on :8000, MinIO on :9000. Next: upload_frames_to_minio.py"
```

Note: `TEMPORAL_API_PROFILE` must reach the `api` service. Add it to the compose `api.environment` list (edit `api/docker-compose.yml`):

```yaml
      - TEMPORAL_API_S3_REGION=us-east-1
      - TEMPORAL_API_PROFILE=${TEMPORAL_API_PROFILE:-false}
```

- [ ] **Step 2: Create `upload_frames_to_minio.py`**

```python
#!/usr/bin/env python
"""Upload the pyro-annotator frames to the VM's MinIO `frames` bucket.

Keys = each frame's path relative to the store root — the exact keys the
benchmark client POSTs. Idempotent: skips objects that already exist.

Usage: uv run python scripts/upload_frames_to_minio.py \
    --store data/03_primary/sequences --endpoint http://localhost:9000
"""

import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from temporal_model.benchmark.dataset import iter_sequences
from temporal_model.benchmark.run_api import frame_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    ap.add_argument("--endpoint", default="http://localhost:9000")
    ap.add_argument("--bucket", default="frames")
    ap.add_argument("--access-key", default="minioadmin")
    ap.add_argument("--secret-key", default="minioadmin")
    args = ap.parse_args()

    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        region_name="us-east-1",
    )
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError:
        s3.create_bucket(Bucket=args.bucket)

    uploaded = skipped = 0
    for seq in iter_sequences(args.store):
        for f in seq.frames:
            key = frame_key(args.store, f)
            try:
                s3.head_object(Bucket=args.bucket, Key=key)
                skipped += 1
                continue
            except ClientError:
                pass
            s3.upload_file(str(f.image_path), args.bucket, key)
            uploaded += 1
        if (uploaded + skipped) % 500 == 0:
            print(f"... {uploaded} uploaded, {skipped} skipped")
    print(f"done: {uploaded} uploaded, {skipped} skipped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable + syntax-check**

Run:
```bash
cd benchmark && chmod +x scripts/provision_api_vm.sh
bash -n scripts/provision_api_vm.sh && echo "provision_api_vm.sh ok"
uv run python -c "import ast; ast.parse(open('scripts/upload_frames_to_minio.py').read()); print('upload script parses')"
```
Expected: both `ok` / `parses`.

- [ ] **Step 4: Commit**

```bash
cd .. && git add benchmark/scripts/provision_api_vm.sh benchmark/scripts/upload_frames_to_minio.py api/docker-compose.yml
git commit -m "feat(benchmark): API e2e provisioning — docker compose + MinIO frame upload"
```

---

## Task 8: README + spec sync

**Files:**
- Modify: `benchmark/README.md`

- [ ] **Step 1: Document the API path**

In `benchmark/README.md`, change the scope note to mark Phase 2 done, and add a section after the core "Run" section:

```markdown
## API end-to-end benchmark (Phase 2)

Measures the real serving path (HTTP → S3 fetch → cached detection → classifier
→ serialization) on the VM, with server-side per-stage timing via
`TEMPORAL_API_PROFILE`. Two passes: **cold** (each full sequence once,
worst-case) and **warm** (growing prefixes per sequence, steady-state with the
detection cache amortizing the detector).

```bash
# on the VM: bring up API + MinIO (profiling on) and upload frames
scripts/provision_api_vm.sh ubuntu@<host>
uv run python scripts/upload_frames_to_minio.py --store data/03_primary/sequences

# run the benchmark against the local API
uv run temporal-benchmark api --url http://localhost:8000 --store data/03_primary/sequences
```

Writes `data/08_reporting/<host>-api-<timestamp>/` (raw.parquet, summary.json,
report.md) with cold/warm e2e latency, stage breakdown, and cache hit rate.
```

Also update the scope blockquote near the top: change "The API end-to-end path … is Phase 2 and not yet implemented." to "The API end-to-end path is implemented — see *API end-to-end benchmark* below."

- [ ] **Step 2: Commit**

```bash
git add benchmark/README.md
git commit -m "docs(benchmark): document the API e2e (Phase 2) workflow"
```

---

## Self-Review

**Spec coverage:**
- §1 server-side profiling (setting, timer threading, s3_fetch/detector/stages, cache counts, logged + details.profiling) → Tasks 1–3 ✓
- §2 run_api cold/warm client → Task 4 ✓
- §3 reporting (summarize_api cold/warm, hit rate, write_api_report) → Task 5 ✓
- §4 CLI `api` subcommand → Task 6 ✓
- §5 provisioning (docker+MinIO, upload script) → Task 7 ✓
- Testing (api profiling on/off; summarize_api) → Tasks 1–3, 5 ✓
- README/docs → Task 8 ✓

**Type/signature consistency:** `to_response(profiling=...)` (Task 1) matches the app call (Task 3). `ModelRunner.predict(paths, *, timer=None, profile=None)` (Task 2) matches the app call (Task 3). `run_api(store_dir, base_url, *, passes, warmup, limit, warm_min_frames)` (Task 4) matches the CLI call (Task 6). `frame_key`/`build_requests`/`rows_for_sequence` names consistent between `run_api.py`, its test, and `upload_frames_to_minio.py`. `summarize_api`/`write_api_report` (Task 5) match the CLI call (Task 6). Profiling payload keys (`stages_ms`, `total_ms`, `n_frames`, `cache_hits`, `cache_misses`) consistent across app (Task 3), run_api (Task 4), and tests.

**Placeholder scan:** none — every step has concrete code and commands.
