# Benchmark Phase 1 (core path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `temporal-model-benchmark` package that runs the core model's `predict()` in-process over the pyro-annotator sequence store, times each pipeline stage, and emits a raw per-sequence table, an aggregate JSON summary, plots, and a markdown report — runnable natively on a CPU VM.

**Architecture:** A shared `StageTimer` added to `core` lets `predict()` record per-stage wall-clock ms only when a timer is passed (no-op and bit-for-bit identical otherwise). The new `benchmark` package loads the `meta.json` sequence store into `core` `Frame`s, drives `predict(frames, timer=…)` with a background resource sampler, and renders artifacts into a self-describing results dir stamped with machine metadata.

**Tech Stack:** Python 3.12 (via uv), PyTorch (CPU), pandas + pyarrow (parquet), matplotlib (Agg), psutil (+ optional pynvml), argparse CLI.

**Spec:** `docs/specs/2026-06-09-benchmark-package-design.md` (Phase 1 only; API e2e path §6/§7 are Phase 2 and out of scope here).

---

## File Structure

**Core (one change):**
- Create `core/src/temporal_model/core/stage_timer.py` — `StageTimer` + `stage_ctx` helper.
- Modify `core/src/temporal_model/core/model.py` — wrap the six stages of `predict()` in `stage_ctx(timer, …)`; add `*, timer=None` kwarg.
- Create `core/tests/test_stage_timer.py` — pure `StageTimer` unit tests.
- Modify `core/tests/test_model_edge_cases.py` — one test that `predict(timer=…)` populates stage timings.

**New package `benchmark/`:**
- `pyproject.toml`, `Makefile`, `README.md`, `.dvc/config`, `.dvcignore`, `.gitignore`
- `src/temporal_model/benchmark/__init__.py`
- `src/temporal_model/benchmark/dataset.py` — `meta.json` store → `BenchSequence`
- `src/temporal_model/benchmark/machine.py` — host/CPU/GPU/torch metadata
- `src/temporal_model/benchmark/resources.py` — background CPU/RAM(/GPU) sampler
- `src/temporal_model/benchmark/run_core.py` — drive `predict()`, collect raw rows
- `src/temporal_model/benchmark/report.py` — aggregate → summary.json + plots + report.md
- `src/temporal_model/benchmark/cli.py` — `temporal-benchmark core`
- `scripts/provision_vm.sh`, `scripts/push_data.sh`, `scripts/pull_results.sh`
- `tests/test_dataset.py`, `tests/test_machine.py`, `tests/test_resources.py`, `tests/test_report.py`
- `data/sequences/pyro-annotator/` (DVC-tracked)

**Root:**
- Modify `Makefile` — add `benchmark` to `PACKAGES`.

---

## Task 1: Core `StageTimer`

**Files:**
- Create: `core/src/temporal_model/core/stage_timer.py`
- Test: `core/tests/test_stage_timer.py`

- [ ] **Step 1: Write the failing test**

```python
# core/tests/test_stage_timer.py
"""Unit tests for the StageTimer profiling helper."""

import time
from contextlib import nullcontext

from temporal_model.core.stage_timer import StageTimer, stage_ctx


def test_records_stage_duration_in_ms():
    timer = StageTimer()
    with timer.stage("yolo"):
        time.sleep(0.01)
    timings = timer.as_dict()
    assert set(timings) == {"yolo"}
    assert timings["yolo"] >= 9.0  # ~10ms, allow scheduling slack


def test_accumulates_repeated_stage():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("vit"):
            time.sleep(0.005)
    assert timer.as_dict()["vit"] >= 12.0  # 3 * ~5ms


def test_as_dict_returns_a_copy():
    timer = StageTimer()
    with timer.stage("crop"):
        pass
    snapshot = timer.as_dict()
    snapshot["crop"] = -1.0
    assert timer.as_dict()["crop"] != -1.0


def test_cpu_timer_does_not_flag_cuda():
    timer = StageTimer(device="cpu")
    assert timer._cuda is False


def test_stage_ctx_is_noop_without_timer():
    ctx = stage_ctx(None, "yolo")
    assert isinstance(ctx, nullcontext)


def test_stage_ctx_delegates_to_timer():
    timer = StageTimer()
    with stage_ctx(timer, "tubes"):
        pass
    assert "tubes" in timer.as_dict()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_stage_timer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_model.core.stage_timer'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/src/temporal_model/core/stage_timer.py
"""Optional per-stage wall-clock profiling for BboxTubeTemporalModel.predict().

A StageTimer is threaded into ``predict()`` only when profiling is requested.
When no timer is passed the prediction path uses ``nullcontext`` and is
bit-for-bit identical to the unprofiled path — no timing, no CUDA syncs.

On a CUDA device the timer synchronises at each stage boundary so GPU stage
times reflect real kernel completion rather than launch latency. These syncs
run only while profiling is active.
"""

import time
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


class StageTimer:
    """Accumulates per-stage wall-clock durations in milliseconds."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        dev = torch.device(device) if device is not None else None
        self._cuda = dev is not None and dev.type == "cuda"
        self._timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[name] = self._timings.get(name, 0.0) + elapsed_ms

    def as_dict(self) -> dict[str, float]:
        return dict(self._timings)


def stage_ctx(timer: StageTimer | None, name: str):
    """Return ``timer.stage(name)`` or a no-op context when ``timer`` is None."""
    return timer.stage(name) if timer is not None else nullcontext()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd core && uv run pytest tests/test_stage_timer.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd core && uv run ruff check src/temporal_model/core/stage_timer.py tests/test_stage_timer.py && uv run ruff format src/temporal_model/core/stage_timer.py tests/test_stage_timer.py
cd .. && git add core/src/temporal_model/core/stage_timer.py core/tests/test_stage_timer.py
git commit -m "feat(core): add opt-in StageTimer for per-stage profiling"
```

---

## Task 2: Wire `timer` into `predict()`

Wrap each of the six pipeline stages in `stage_ctx(timer, …)`. When `timer is None` every wrapper is `nullcontext()`, so control flow and numerics are unchanged — the existing suite (including `test_model_parity.py`) must keep passing.

**Files:**
- Modify: `core/src/temporal_model/core/model.py`
- Test: `core/tests/test_model_edge_cases.py`

- [ ] **Step 1: Add the failing integration test** (reuses existing fixtures in this file: `tiny_classifier`, `red_frames`, `_fake_yolo_factory`, `TEST_CONFIG`)

Append to `core/tests/test_model_edge_cases.py`:

```python
class TestStageTimerIntegration:
    def test_predict_populates_all_stage_timings(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        from temporal_model.core.stage_timer import StageTimer

        # One stable box per frame -> a tube survives -> every stage runs.
        boxes = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(boxes)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        timer = StageTimer()
        model.predict(frames=red_frames, timer=timer)
        timings = timer.as_dict()
        assert {"pad", "yolo", "tubes", "crop", "vit", "trigger"} <= set(timings)
        assert all(v >= 0.0 for v in timings.values())

    def test_predict_without_timer_is_unaffected(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        boxes = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
        yolo = _fake_yolo_factory(boxes)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)  # no timer kwarg
        assert isinstance(out.is_positive, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd core && uv run pytest tests/test_model_edge_cases.py::TestStageTimerIntegration -v`
Expected: FAIL — `predict() got an unexpected keyword argument 'timer'`

- [ ] **Step 3: Implement — add the import and the `timer` kwarg**

In `core/src/temporal_model/core/model.py`, add to the import block (after the `.protocol` import line):

```python
from .stage_timer import StageTimer, stage_ctx
```

Change the `predict` signature (line ~127) from:

```python
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
```
to:
```python
    def predict(
        self, frames: list[Frame], *, timer: StageTimer | None = None
    ) -> TemporalModelOutput:
```

- [ ] **Step 4: Implement — wrap the six stages**

Each edit only adds a `with stage_ctx(timer, "<name>"):` wrapper and indents the existing block one level. Do not change any logic inside.

**(a) `pad` stage** — wrap the truncate+pad block. Replace:

```python
        truncated = frames[: clf_cfg["max_frames"]]
        n_truncated = original_len - len(truncated)

        padded_indices: list[int] = []
        pad_min = int(infer.get("pad_to_min_frames", 0))
        if pad_min > 0 and len(truncated) < pad_min:
            strategy = infer.get("pad_strategy", "symmetric")
            try:
                pad_fn = _PAD_STRATEGIES[strategy]
            except KeyError as e:
                raise ValueError(
                    f"unknown pad_strategy {strategy!r}; "
                    f"expected one of {sorted(_PAD_STRATEGIES)}"
                ) from e
            truncated, padded_indices = pad_fn(truncated, min_length=pad_min)
```
with:
```python
        with stage_ctx(timer, "pad"):
            truncated = frames[: clf_cfg["max_frames"]]
            n_truncated = original_len - len(truncated)

            padded_indices: list[int] = []
            pad_min = int(infer.get("pad_to_min_frames", 0))
            if pad_min > 0 and len(truncated) < pad_min:
                strategy = infer.get("pad_strategy", "symmetric")
                try:
                    pad_fn = _PAD_STRATEGIES[strategy]
                except KeyError as e:
                    raise ValueError(
                        f"unknown pad_strategy {strategy!r}; "
                        f"expected one of {sorted(_PAD_STRATEGIES)}"
                    ) from e
                truncated, padded_indices = pad_fn(truncated, min_length=pad_min)
```

**(b) `yolo` stage** — replace:

```python
        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )
```
with:
```python
        with stage_ctx(timer, "yolo"):
            frame_dets = run_yolo_on_frames(
                self._yolo,
                truncated,
                confidence_threshold=infer["confidence_threshold"],
                iou_nms=infer["iou_nms"],
                image_size=infer["image_size"],
                device=self._device,
            )
```

**(c) `tubes` stage** — replace the candidate + kept block:

```python
        # Pre-merge (raw) candidates count, for the details JSON.
        candidate_tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )
        kept = build_tubes_for_inference(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
            min_tube_length=tubes_cfg["infer_min_tube_length"],
            min_detected_entries=tubes_cfg["min_detected_entries"],
            interpolate_gaps=tubes_cfg["interpolate_gaps"],
            merge_iomin=tubes_cfg.get("merge_iomin"),
            merge_prox_factor=tubes_cfg.get("merge_prox_factor"),
            merge_max_gap=tubes_cfg.get("merge_max_gap"),
        )
```
with:
```python
        with stage_ctx(timer, "tubes"):
            # Pre-merge (raw) candidates count, for the details JSON.
            candidate_tubes = build_tubes(
                frame_dets,
                iou_threshold=tubes_cfg["iou_threshold"],
                max_misses=tubes_cfg["max_misses"],
            )
            kept = build_tubes_for_inference(
                frame_dets,
                iou_threshold=tubes_cfg["iou_threshold"],
                max_misses=tubes_cfg["max_misses"],
                min_tube_length=tubes_cfg["infer_min_tube_length"],
                min_detected_entries=tubes_cfg["min_detected_entries"],
                interpolate_gaps=tubes_cfg["interpolate_gaps"],
                merge_iomin=tubes_cfg.get("merge_iomin"),
                merge_prox_factor=tubes_cfg.get("merge_prox_factor"),
                merge_max_gap=tubes_cfg.get("merge_max_gap"),
            )
```

**(d) `crop` stage** — replace the patch loop:

```python
        patches_per_tube: list[torch.Tensor] = []
        masks_per_tube: list[torch.Tensor] = []
        for t in kept:
            p, m = crop_tube_patches(
                t,
                truncated,
                context_factor=mi["context_factor"],
                patch_size=mi["patch_size"],
                max_frames=clf_cfg["max_frames"],
                normalization_mean=mi["normalization"]["mean"],
                normalization_std=mi["normalization"]["std"],
                stabilize=mi.get("stabilize", True),
            )
            patches_per_tube.append(p.to(self._device))
            masks_per_tube.append(m.to(self._device))
```
with:
```python
        patches_per_tube: list[torch.Tensor] = []
        masks_per_tube: list[torch.Tensor] = []
        with stage_ctx(timer, "crop"):
            for t in kept:
                p, m = crop_tube_patches(
                    t,
                    truncated,
                    context_factor=mi["context_factor"],
                    patch_size=mi["patch_size"],
                    max_frames=clf_cfg["max_frames"],
                    normalization_mean=mi["normalization"]["mean"],
                    normalization_std=mi["normalization"]["std"],
                    stabilize=mi.get("stabilize", True),
                )
                patches_per_tube.append(p.to(self._device))
                masks_per_tube.append(m.to(self._device))
```

**(e) `vit` stage** — replace:

```python
        logits = score_tubes(
            self._classifier,
            patches_per_tube=patches_per_tube,
            masks_per_tube=masks_per_tube,
        )
```
with:
```python
        with stage_ctx(timer, "vit"):
            logits = score_tubes(
                self._classifier,
                patches_per_tube=patches_per_tube,
                masks_per_tube=masks_per_tube,
            )
```

**(f) `trigger` stage** — replace:

```python
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
                logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
                min_prefix_length=tubes_cfg["infer_min_tube_length"],
            )
        )
```
with:
```python
        with stage_ctx(timer, "trigger"):
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
                    logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
                    min_prefix_length=tubes_cfg["infer_min_tube_length"],
                )
            )
```

- [ ] **Step 5: Run the new test + the full core suite (parity guard)**

Run: `cd core && uv run pytest tests/test_model_edge_cases.py::TestStageTimerIntegration tests/test_model_parity.py -v`
Expected: PASS (new tests pass; parity unchanged)

Then the whole core suite:
Run: `cd core && uv run pytest tests/ -q`
Expected: PASS (no regressions)

- [ ] **Step 6: Lint + commit**

```bash
cd core && uv run ruff check . && uv run ruff format .
cd .. && git add core/src/temporal_model/core/model.py core/tests/test_model_edge_cases.py
git commit -m "feat(core): time predict() stages when a StageTimer is passed"
```

---

## Task 3: Scaffold the `benchmark` package

**Files:**
- Create: `benchmark/pyproject.toml`, `benchmark/Makefile`, `benchmark/README.md`, `benchmark/.dvc/config`, `benchmark/.dvcignore`, `benchmark/.gitignore`, `benchmark/src/temporal_model/benchmark/__init__.py`
- Modify: `Makefile` (root)

- [ ] **Step 1: Create `benchmark/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-benchmark"
version = "0.1.0"
description = "Latency/throughput/resource benchmark for the temporal smoke classifier"
requires-python = ">=3.11"
dependencies = [
    "temporal-model-core",
    "pandas>=2.2",
    "pyarrow>=15",
    "matplotlib>=3.8",
    "psutil>=5.9",
    "nvidia-ml-py>=12",
]

[project.scripts]
temporal-benchmark = "temporal_model.benchmark.cli:main"

[tool.uv.sources]
temporal-model-core = { path = "../core", editable = true }

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "dvc[s3]>=3.56",
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Create `benchmark/Makefile`**

```makefile
.PHONY: install lint format test

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Create `benchmark/.dvc/config`**

```ini
[core]
    remote = s3remote
    analytics = false
['remote "s3remote"']
    url = s3://pyro-vision-rd/dvc/temporal-model/benchmark/
```

- [ ] **Step 4: Create `benchmark/.dvcignore`** (single line)

```
results/
```

- [ ] **Step 5: Create `benchmark/.gitignore`**

```
.venv/
results/
__pycache__/
/data/sequences/
```

- [ ] **Step 6: Create `benchmark/src/temporal_model/benchmark/__init__.py`** (empty namespace marker)

```python
"""Latency/throughput/resource benchmark for the temporal smoke classifier."""
```

- [ ] **Step 7: Create `benchmark/README.md`**

```markdown
# temporal-model-benchmark

Measures the temporal smoke classifier's latency, throughput, and resource
usage, with a per-stage breakdown of `predict()`. Import as
`temporal_model.benchmark`. Depends on `temporal-model-core`.

Phase 1 covers the **core in-process** path. See
`docs/specs/2026-06-09-benchmark-package-design.md`.

## Run

```bash
make install                       # uv sync (installs core editable)
temporal-benchmark core \
    --store data/sequences \
    --model ../api/models/model.zip \
    --out results
```

Outputs a self-describing dir `results/<host>-<timestamp>/` with `raw.parquet`,
`resources.parquet`, `summary.json`, `plots/*.png`, and `report.md`.

## On a VM

See `scripts/provision_vm.sh`, `scripts/push_data.sh`, `scripts/pull_results.sh`.
```

- [ ] **Step 8: Register the package in the root `Makefile`**

Replace `PACKAGES := core train eval api` with:
```makefile
PACKAGES := core train eval api benchmark
```

- [ ] **Step 9: Sync and verify the package builds**

Run: `cd benchmark && uv sync`
Expected: resolves and installs `temporal-model-core` (editable) plus deps, no errors.

- [ ] **Step 10: Commit**

```bash
git add benchmark/pyproject.toml benchmark/Makefile benchmark/README.md \
        benchmark/.dvc/config benchmark/.dvcignore benchmark/.gitignore \
        benchmark/src/temporal_model/benchmark/__init__.py benchmark/uv.lock Makefile
git commit -m "chore(benchmark): scaffold temporal-model-benchmark package"
```

---

## Task 4: Dataset loader (`dataset.py`)

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/dataset.py`
- Test: `benchmark/tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_dataset.py
"""Tests for the meta.json sequence-store loader."""

import json
from pathlib import Path

from temporal_model.benchmark.dataset import BenchSequence, iter_sequences


def _write_seq(seq_dir: Path, key: str, label: str, files: list[str]) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (seq_dir / f).write_bytes(b"x")
    meta = {
        "key": key,
        "label": label,
        "frames": [{"file": f, "created_at": None} for f in files],
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta))


def test_loads_sequences_in_frame_order(tmp_path: Path):
    seq = tmp_path / "org" / "cam" / "seq_1"
    _write_seq(seq, key="seq_1", label="smoke", files=["a.jpg", "b.jpg", "c.jpg"])

    out = list(iter_sequences(tmp_path))

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, BenchSequence)
    assert s.key == "seq_1"
    assert s.label == "smoke"
    assert s.frame_count == 3
    assert [f.image_path.name for f in s.frames] == ["a.jpg", "b.jpg", "c.jpg"]
    assert s.frames[0].image_path == seq / "a.jpg"


def test_finds_sequences_recursively(tmp_path: Path):
    _write_seq(tmp_path / "a" / "s1", "s1", "fp", ["f0.jpg"])
    _write_seq(tmp_path / "b" / "c" / "s2", "s2", "smoke", ["f0.jpg", "f1.jpg"])
    keys = sorted(s.key for s in iter_sequences(tmp_path))
    assert keys == ["s1", "s2"]


def test_missing_store_yields_nothing(tmp_path: Path):
    assert list(iter_sequences(tmp_path / "nope")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: ...benchmark.dataset`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/src/temporal_model/benchmark/dataset.py
"""Load a meta.json sequence store into core Frame objects.

Reads the same on-disk format the temporal-model-explorer writes
(``meta.json`` with an ordered ``frames`` list), but depends only on
``temporal_model.core`` — no explorer / pyrocore import.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from temporal_model.core.protocol import Frame

META_FILENAME = "meta.json"


@dataclass
class BenchSequence:
    """One benchmarkable sequence: its key, label, and ordered frames."""

    key: str
    label: str
    frame_count: int
    frames: list[Frame]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build(seq_dir: Path, meta: dict) -> BenchSequence:
    frames = [
        Frame(
            frame_id=Path(ref["file"]).stem,
            image_path=seq_dir / ref["file"],
            timestamp=_parse_ts(ref.get("created_at")),
        )
        for ref in meta.get("frames", [])
    ]
    return BenchSequence(
        key=meta.get("key", seq_dir.name),
        label=meta.get("label", "unknown"),
        frame_count=len(frames),
        frames=frames,
    )


def iter_sequences(store_dir: Path) -> Iterator[BenchSequence]:
    """Yield one BenchSequence per meta.json under ``store_dir`` (recursive)."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        meta = json.loads(meta_path.read_text())
        yield _build(meta_path.parent, meta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/dataset.py tests/test_dataset.py && uv run ruff format src/temporal_model/benchmark/dataset.py tests/test_dataset.py
cd .. && git add benchmark/src/temporal_model/benchmark/dataset.py benchmark/tests/test_dataset.py
git commit -m "feat(benchmark): add meta.json sequence-store loader"
```

---

## Task 5: Machine metadata (`machine.py`)

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/machine.py`
- Test: `benchmark/tests/test_machine.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_machine.py
"""Tests for machine metadata capture."""

from temporal_model.benchmark import machine


REQUIRED_KEYS = {
    "hostname", "platform", "cpu_model", "cpu_count_physical",
    "cpu_count_logical", "ram_total_gb", "gpu_name", "torch_version",
    "cuda_version", "python_version", "device", "torch_num_threads",
}


def test_machine_info_has_required_keys():
    info = machine.machine_info(device="cpu")
    assert REQUIRED_KEYS <= set(info)


def test_machine_info_reports_requested_device():
    info = machine.machine_info(device="cpu")
    assert info["device"] == "cpu"


def test_gpu_name_none_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = machine.machine_info(device="cpu")
    assert info["gpu_name"] is None
    assert info["cuda_version"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_machine.py -v`
Expected: FAIL — `ModuleNotFoundError: ...benchmark.machine`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/src/temporal_model/benchmark/machine.py
"""Capture host + runtime metadata so every result dir is self-describing."""

import platform
import socket
import sys

import psutil
import torch


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine_info(*, device: str) -> dict:
    """Return a flat dict of host/CPU/GPU/runtime facts for this run."""
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda else None,
        "python_version": sys.version.split()[0],
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_machine.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/machine.py tests/test_machine.py && uv run ruff format src/temporal_model/benchmark/machine.py tests/test_machine.py
cd .. && git add benchmark/src/temporal_model/benchmark/machine.py benchmark/tests/test_machine.py
git commit -m "feat(benchmark): capture machine metadata"
```

---

## Task 6: Resource sampler (`resources.py`)

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/resources.py`
- Test: `benchmark/tests/test_resources.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_resources.py
"""Tests for the background resource sampler."""

import time

from temporal_model.benchmark.resources import ResourceSampler


def test_collects_samples_while_active():
    with ResourceSampler(interval=0.02) as sampler:
        time.sleep(0.1)
    timeline = sampler.timeline()
    assert len(timeline) >= 2
    sample = timeline[0]
    assert {"t", "cpu_pct", "ram_gb"} <= set(sample)


def test_peaks_present_for_cpu_and_ram():
    with ResourceSampler(interval=0.02) as sampler:
        time.sleep(0.06)
    peaks = sampler.peaks()
    assert "cpu_pct" in peaks
    assert "ram_gb" in peaks


def test_no_samples_before_start():
    sampler = ResourceSampler(interval=0.02)
    assert sampler.timeline() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_resources.py -v`
Expected: FAIL — `ModuleNotFoundError: ...benchmark.resources`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/src/temporal_model/benchmark/resources.py
"""Background sampler for CPU/RAM (always) and GPU (when NVML is available).

Runs a daemon thread that snapshots utilisation every ``interval`` seconds
between ``__enter__`` and ``__exit__``. GPU metrics are best-effort: if the
``pynvml`` bindings or an NVIDIA device are absent, the sampler silently omits
them (CPU-only VMs are fully supported).
"""

import threading
import time

import psutil

try:  # best-effort GPU support
    import pynvml

    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001 — any failure means "no GPU metrics"
    pynvml = None
    _GPU_HANDLE = None


class ResourceSampler:
    """Context manager that records a utilisation timeline."""

    def __init__(self, interval: float = 0.1) -> None:
        self._interval = interval
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "ResourceSampler":
        psutil.cpu_percent(None)  # prime the interval baseline
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample(self) -> dict:
        row = {
            "t": time.perf_counter() - self._t0,
            "cpu_pct": psutil.cpu_percent(None),
            "ram_gb": psutil.virtual_memory().used / 1e9,
        }
        if _GPU_HANDLE is not None:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            row["gpu_util"] = float(util.gpu)
            row["gpu_mem_gb"] = mem.used / 1e9
        return row

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._sample())
            self._stop.wait(self._interval)

    def timeline(self) -> list[dict]:
        return list(self._samples)

    def peaks(self) -> dict:
        if not self._samples:
            return {}
        keys = {k for s in self._samples for k in s if k != "t"}
        return {k: max(s.get(k, 0.0) for s in self._samples) for k in keys}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_resources.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/resources.py tests/test_resources.py && uv run ruff format src/temporal_model/benchmark/resources.py tests/test_resources.py
cd .. && git add benchmark/src/temporal_model/benchmark/resources.py benchmark/tests/test_resources.py
git commit -m "feat(benchmark): add background CPU/RAM/GPU sampler"
```

---

## Task 7: Core-path runner (`run_core.py`)

Drives `predict(frames, timer=…)` over the store and returns a raw DataFrame. Heavy (needs a real model), so verification is a manual smoke run in Task 9, not a unit test — matching repo convention for model-dependent code.

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/run_core.py`

- [ ] **Step 1: Write the implementation**

```python
# benchmark/src/temporal_model/benchmark/run_core.py
"""Run BboxTubeTemporalModel.predict() over a sequence store, timing each stage."""

import logging
from pathlib import Path

import pandas as pd
import torch

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.stage_timer import StageTimer

from .dataset import BenchSequence, iter_sequences

logger = logging.getLogger(__name__)

STAGES = ("pad", "yolo", "tubes", "crop", "vit", "trigger")


def resolve_device(requested: str) -> str:
    """Map ``auto`` to cuda-if-available else cpu; pass anything else through."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _one_rep(model: BboxTubeTemporalModel, seq: BenchSequence, device: str) -> dict:
    timer = StageTimer(device=device)
    output = model.predict(seq.frames, timer=timer)
    timings = timer.as_dict()
    row = {
        "key": seq.key,
        "label": seq.label,
        "frame_count": seq.frame_count,
        "n_kept_tubes": len(output.details.get("tubes", {}).get("kept", [])),
        "is_positive": output.is_positive,
    }
    for stage in STAGES:
        row[f"{stage}_ms"] = timings.get(stage, 0.0)
    row["total_ms"] = sum(timings.get(s, 0.0) for s in STAGES)
    return row


def run_core(
    store_dir: Path,
    model_path: Path,
    *,
    device: str = "auto",
    reps: int = 5,
    warmup: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """Benchmark predict() over every sequence; one row per (sequence, rep)."""
    device = resolve_device(device)
    model = BboxTubeTemporalModel.from_package(model_path, device=device)

    sequences = list(iter_sequences(store_dir))
    if limit is not None:
        sequences = sequences[:limit]
    if not sequences:
        raise SystemExit(f"no sequences found under {store_dir}")

    logger.info("warming up on %d sequences", min(warmup, len(sequences)))
    for seq in sequences[:warmup]:
        model.predict(seq.frames)

    rows: list[dict] = []
    for i, seq in enumerate(sequences):
        for rep in range(reps):
            try:
                row = _one_rep(model, seq, device)
                row["rep"] = rep
                row["failed"] = False
            except Exception as exc:  # noqa: BLE001 — record + continue
                logger.warning("sequence %s failed: %s", seq.key, exc)
                row = {"key": seq.key, "rep": rep, "failed": True}
            rows.append(row)
        if (i + 1) % 25 == 0:
            logger.info("benchmarked %d/%d sequences", i + 1, len(sequences))

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Verify it imports**

Run: `cd benchmark && uv run python -c "from temporal_model.benchmark.run_core import run_core, resolve_device; print(resolve_device('auto'))"`
Expected: prints `cpu` (or `cuda` on a GPU box), no import error.

- [ ] **Step 3: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/run_core.py && uv run ruff format src/temporal_model/benchmark/run_core.py
cd .. && git add benchmark/src/temporal_model/benchmark/run_core.py
git commit -m "feat(benchmark): add core-path runner with per-stage timing"
```

---

## Task 8: Report / aggregation (`report.py`)

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/report.py`
- Test: `benchmark/tests/test_report.py`

- [ ] **Step 1: Write the failing test** (covers the pure aggregation; plot/markdown rendering is exercised by the smoke run in Task 9)

```python
# benchmark/tests/test_report.py
"""Tests for benchmark aggregation."""

import pandas as pd

from temporal_model.benchmark.report import summarize

STAGES = ["pad", "yolo", "tubes", "crop", "vit", "trigger"]


def _row(key, total, **stage_ms):
    r = {"key": key, "rep": 0, "failed": False, "frame_count": 6,
         "n_kept_tubes": 1, "total_ms": total}
    for s in STAGES:
        r[f"{s}_ms"] = stage_ms.get(s, 0.0)
    return r


def test_summarize_latency_percentiles_and_counts():
    df = pd.DataFrame([
        _row("a", 100.0, vit=80.0, yolo=20.0),
        _row("b", 200.0, vit=160.0, yolo=40.0),
        _row("c", 300.0, vit=240.0, yolo=60.0),
    ])
    s = summarize(df)
    assert s["n_sequences"] == 3
    assert s["n_failed"] == 0
    assert s["total_ms"]["p50"] == 200.0
    # frames/sec uses mean latency over total frames; just assert it's positive.
    assert s["throughput"]["sequences_per_sec"] > 0
    # vit dominates the mean stage share.
    assert s["stage_share_pct"]["vit"] > s["stage_share_pct"]["yolo"]


def test_summarize_counts_failures_and_excludes_them():
    df = pd.DataFrame([
        _row("a", 100.0, vit=100.0),
        {"key": "b", "rep": 0, "failed": True},
    ])
    s = summarize(df)
    assert s["n_sequences"] == 2
    assert s["n_failed"] == 1
    assert s["total_ms"]["p50"] == 100.0  # failed row excluded from latency
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd benchmark && uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: ...benchmark.report`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/src/temporal_model/benchmark/report.py
"""Aggregate the raw benchmark table into a summary, plots, and a markdown report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

STAGES = ["pad", "yolo", "tubes", "crop", "vit", "trigger"]


def _pct(series: pd.Series, q: float) -> float:
    return round(float(series.quantile(q)), 3)


def summarize(df: pd.DataFrame) -> dict:
    """Compute latency percentiles, throughput, and mean stage shares."""
    ok = df[~df["failed"]] if "failed" in df else df
    total = ok["total_ms"]
    mean_total_ms = float(total.mean()) if len(total) else 0.0
    mean_frames = float(ok["frame_count"].mean()) if len(ok) else 0.0
    seq_per_sec = 1000.0 / mean_total_ms if mean_total_ms else 0.0

    stage_means = {s: float(ok[f"{s}_ms"].mean()) for s in STAGES} if len(ok) else {
        s: 0.0 for s in STAGES
    }
    stage_total = sum(stage_means.values()) or 1.0

    return {
        "n_sequences": int(df["key"].nunique()),
        "n_failed": int(df["failed"].sum()) if "failed" in df else 0,
        "total_ms": {
            "p50": _pct(total, 0.50),
            "p90": _pct(total, 0.90),
            "p99": _pct(total, 0.99),
            "mean": round(mean_total_ms, 3),
        },
        "stage_ms_mean": {s: round(v, 3) for s, v in stage_means.items()},
        "stage_share_pct": {
            s: round(100.0 * v / stage_total, 1) for s, v in stage_means.items()
        },
        "throughput": {
            "sequences_per_sec": round(seq_per_sec, 3),
            "frames_per_sec": round(seq_per_sec * mean_frames, 3),
        },
    }


def _plot_latency_hist(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots()
    df.loc[~df["failed"], "total_ms"].plot.hist(bins=30, ax=ax)
    ax.set_xlabel("total latency (ms)")
    ax.set_title("Per-sequence latency distribution")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_stage_breakdown(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots()
    means = summary["stage_ms_mean"]
    ax.bar(list(means), list(means.values()))
    ax.set_ylabel("mean ms")
    ax.set_title("Mean time per stage")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_vs_frames(df: pd.DataFrame, out: Path) -> None:
    ok = df[~df["failed"]]
    fig, ax = plt.subplots()
    ax.scatter(ok["frame_count"], ok["total_ms"], s=8, alpha=0.5)
    ax.set_xlabel("frame count")
    ax.set_ylabel("total latency (ms)")
    ax.set_title("Latency vs frame count")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_resources(resources: pd.DataFrame, out: Path) -> None:
    if resources.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(resources["t"], resources["cpu_pct"], label="CPU %")
    if "gpu_util" in resources:
        ax.plot(resources["t"], resources["gpu_util"], label="GPU %")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("utilisation %")
    ax.set_title("Resource utilisation")
    ax.legend()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    resources: pd.DataFrame,
    machine: dict,
    out_dir: Path,
) -> dict:
    """Write raw.parquet, resources.parquet, summary.json, plots, report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    df.to_parquet(out_dir / "raw.parquet")
    resources.to_parquet(out_dir / "resources.parquet")

    summary = summarize(df)
    summary["machine"] = machine
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_latency_hist(df, plots / "latency_hist.png")
    _plot_stage_breakdown(summary, plots / "stage_breakdown.png")
    _plot_latency_vs_frames(df, plots / "latency_vs_frames.png")
    _plot_resources(resources, plots / "resources.png")

    (out_dir / "report.md").write_text(_render_markdown(summary))
    return summary


def _render_markdown(summary: dict) -> str:
    m = summary["machine"]
    lat = summary["total_ms"]
    tp = summary["throughput"]
    lines = [
        f"# Benchmark report — {m['hostname']}",
        "",
        "## Machine",
        f"- CPU: {m['cpu_model']} ({m['cpu_count_physical']} phys / "
        f"{m['cpu_count_logical']} logical, {m['torch_num_threads']} torch threads)",
        f"- RAM: {m['ram_total_gb']} GB",
        f"- GPU: {m['gpu_name'] or 'none'}",
        f"- device: {m['device']} · torch {m['torch_version']} · "
        f"python {m['python_version']}",
        "",
        "## Latency (total, ms)",
        f"- p50 {lat['p50']} · p90 {lat['p90']} · p99 {lat['p99']} · "
        f"mean {lat['mean']}",
        "",
        "## Throughput",
        f"- {tp['sequences_per_sec']} seq/s · {tp['frames_per_sec']} frames/s",
        "",
        "## Stage breakdown (mean ms · share)",
    ]
    for stage in STAGES:
        ms = summary["stage_ms_mean"][stage]
        pct = summary["stage_share_pct"][stage]
        lines.append(f"- {stage}: {ms} ms ({pct}%)")
    lines += [
        "",
        f"Sequences: {summary['n_sequences']} · failed: {summary['n_failed']}",
        "",
        "## Plots",
        "![latency](plots/latency_hist.png)",
        "![stages](plots/stage_breakdown.png)",
        "![latency vs frames](plots/latency_vs_frames.png)",
        "![resources](plots/resources.png)",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd benchmark && uv run pytest tests/test_report.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/report.py tests/test_report.py && uv run ruff format src/temporal_model/benchmark/report.py tests/test_report.py
cd .. && git add benchmark/src/temporal_model/benchmark/report.py benchmark/tests/test_report.py
git commit -m "feat(benchmark): aggregate summary, plots, and markdown report"
```

---

## Task 9: CLI (`cli.py`) + end-to-end smoke run

**Files:**
- Create: `benchmark/src/temporal_model/benchmark/cli.py`

- [ ] **Step 1: Write the implementation**

```python
# benchmark/src/temporal_model/benchmark/cli.py
"""`temporal-benchmark` CLI. Phase 1 implements the `core` subcommand."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from .machine import machine_info
from .report import write_report
from .resources import ResourceSampler
from .run_core import resolve_device, run_core


def _run_core_cmd(args: argparse.Namespace) -> None:
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    with ResourceSampler(interval=args.sample_interval) as sampler:
        df = run_core(
            args.store,
            args.model,
            device=device,
            reps=args.reps,
            warmup=args.warmup,
            limit=args.limit,
        )
    resources = pd.DataFrame(sampler.timeline())

    stamp = args.timestamp
    out_dir = args.out / f"{machine_info(device=device)['hostname']}-{stamp}"
    summary = write_report(df, resources, machine_info(device=device), out_dir)

    print(f"wrote {out_dir}")
    print(
        f"  p50 {summary['total_ms']['p50']}ms · "
        f"{summary['throughput']['sequences_per_sec']} seq/s · "
        f"{summary['n_failed']} failed"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="temporal-benchmark")
    sub = ap.add_subparsers(dest="command", required=True)

    core = sub.add_parser("core", help="in-process predict() stage breakdown")
    core.add_argument("--store", type=Path, required=True)
    core.add_argument("--model", type=Path, required=True)
    core.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    core.add_argument("--reps", type=int, default=5)
    core.add_argument("--warmup", type=int, default=3)
    core.add_argument("--limit", type=int, default=None)
    core.add_argument("--threads", type=int, default=None,
                      help="torch.set_num_threads(); default = torch default")
    core.add_argument("--sample-interval", type=float, default=0.1)
    core.add_argument("--out", type=Path, default=Path("results"))
    core.add_argument(
        "--timestamp",
        default="run",
        help="label appended to the results dir (e.g. 20260609-1530)",
    )
    core.set_defaults(func=_run_core_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fetch the model** (from repo root)

Run: `make fetch-model`
Expected: writes `api/models/model.zip` (downloads v0.2.0 from HuggingFace, no creds).

- [ ] **Step 3: Smoke-run the CLI on a few sequences** (needs the local sequence store; point `--store` at the explorer's pyro-annotator copy until Task 11 lands the in-repo data)

Run:
```bash
cd benchmark && uv run temporal-benchmark core \
    --store /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/temporal-models/temporal-model-explorer/data/03_primary/sequences/pyro-annotator \
    --model ../api/models/model.zip \
    --reps 2 --warmup 1 --limit 5 --timestamp smoke --out results
```
Expected: prints `wrote results/<host>-smoke` plus a p50/throughput line; the dir contains `raw.parquet`, `resources.parquet`, `summary.json`, `plots/*.png`, `report.md`. Open `report.md` and confirm the stage breakdown is populated (vit/yolo non-zero).

- [ ] **Step 4: Lint + commit** (results/ is gitignored)

```bash
cd benchmark && uv run ruff check src/temporal_model/benchmark/cli.py && uv run ruff format src/temporal_model/benchmark/cli.py
cd .. && git add benchmark/src/temporal_model/benchmark/cli.py
git commit -m "feat(benchmark): add temporal-benchmark core CLI"
```

---

## Task 10: VM helper scripts

**Files:**
- Create: `benchmark/scripts/provision_vm.sh`, `benchmark/scripts/push_data.sh`, `benchmark/scripts/pull_results.sh`

- [ ] **Step 1: Create `benchmark/scripts/provision_vm.sh`**

```bash
#!/usr/bin/env bash
# Bootstrap a fresh VM to run the core benchmark: uv, repo, python 3.12, deps, model.
# Usage: provision_vm.sh <ssh-host> [repo-url]
set -euo pipefail
HOST="${1:?usage: provision_vm.sh <ssh-host> [repo-url]}"
REPO="${2:-https://github.com/pyronear/temporal-model.git}"

ssh "$HOST" "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
ssh "$HOST" "test -d temporal-model || git clone '$REPO' temporal-model"
ssh "$HOST" "cd temporal-model && \$HOME/.local/bin/uv python install 3.12 && \
             make -C benchmark install && make fetch-model"
echo "provisioned $HOST — now run scripts/push_data.sh $HOST"
```

- [ ] **Step 2: Create `benchmark/scripts/push_data.sh`**

```bash
#!/usr/bin/env bash
# rsync the local pyro-annotator sequence store to the VM (no S3 creds needed there).
# Run from the benchmark/ dir. Usage: push_data.sh <ssh-host>
set -euo pipefail
HOST="${1:?usage: push_data.sh <ssh-host>}"
SRC="data/sequences/pyro-annotator/"
DST="$HOST:~/temporal-model/benchmark/data/sequences/pyro-annotator/"
test -d "$SRC" || { echo "missing $SRC (run 'dvc pull' first)"; exit 1; }
rsync -az --info=progress2 "$SRC" "$DST"
echo "pushed dataset to $HOST"
```

- [ ] **Step 3: Create `benchmark/scripts/pull_results.sh`**

```bash
#!/usr/bin/env bash
# rsync benchmark results back from the VM. Run from benchmark/. Usage: pull_results.sh <ssh-host>
set -euo pipefail
HOST="${1:?usage: pull_results.sh <ssh-host>}"
rsync -az --info=progress2 \
    "$HOST:~/temporal-model/benchmark/results/" "./results/"
echo "pulled results from $HOST into ./results/"
```

- [ ] **Step 4: Make executable + verify they parse**

Run:
```bash
chmod +x benchmark/scripts/*.sh
for s in benchmark/scripts/*.sh; do bash -n "$s" && echo "ok: $s"; done
```
Expected: `ok:` for all three (syntax check only — does not connect).

- [ ] **Step 5: Commit**

```bash
git add benchmark/scripts/provision_vm.sh benchmark/scripts/push_data.sh benchmark/scripts/pull_results.sh
git commit -m "feat(benchmark): add VM provision/push/pull helper scripts"
```

---

## Task 11: DVC-track the pyro-annotator dataset

Track the realistic benchmark dataset in this package's DVC remote so any developer can `dvc pull` it and `push_data.sh` it onto a VM.

**Files:**
- Create: `benchmark/data/sequences/pyro-annotator.dvc` (the data itself is gitignored; only the `.dvc` pointer is committed)

- [ ] **Step 1: Initialise DVC in the package** (if not already from scaffolding)

Run:
```bash
cd benchmark && uv run dvc init --subdir 2>/dev/null || true
uv run dvc remote list
```
Expected: lists `s3remote  s3://pyro-vision-rd/dvc/temporal-model/benchmark/`.

- [ ] **Step 2: Copy the pyro-annotator source into the package**

Run:
```bash
cd benchmark && mkdir -p data/sequences
rsync -a \
  /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/temporal-models/temporal-model-explorer/data/03_primary/sequences/pyro-annotator \
  data/sequences/
```
Expected: `data/sequences/pyro-annotator/` exists (~562 MB, 332 `meta.json` files).

Verify: `find data/sequences/pyro-annotator -name meta.json | wc -l` → `332`.

- [ ] **Step 3: Track it with DVC**

Run:
```bash
cd benchmark && uv run dvc add data/sequences/pyro-annotator
```
Expected: creates `data/sequences/pyro-annotator.dvc` and a `.gitignore` entry for the data dir.

- [ ] **Step 4: Push to the remote** (needs S3 write creds in the environment)

Run: `cd benchmark && uv run dvc push`
Expected: uploads the dataset to `s3://pyro-vision-rd/dvc/temporal-model/benchmark/`.

- [ ] **Step 5: Commit the pointer**

```bash
git add benchmark/data/sequences/pyro-annotator.dvc benchmark/data/.gitignore benchmark/.dvc
git commit -m "data(benchmark): track pyro-annotator sequence store via DVC"
```

- [ ] **Step 6: Re-run the smoke test against the in-repo store** (confirms the default `--store data/sequences` works)

Run:
```bash
cd benchmark && uv run temporal-benchmark core \
    --store data/sequences --model ../api/models/model.zip \
    --reps 2 --warmup 1 --limit 5 --timestamp smoke2 --out results
```
Expected: `wrote results/<host>-smoke2` with a populated `report.md`.

---

## Self-Review

**Spec coverage (Phase 1 sections):**
- §1 core stage timer → Tasks 1–2 ✓
- §2 dataset loader → Task 4 ✓
- §3 machine metadata → Task 5 ✓
- §4 resource sampler → Task 6 ✓
- §5 core-path runner → Task 7 ✓
- §8 report/aggregation (raw.parquet, summary.json, plots, report.md) → Task 8 ✓
- §9 CLI (`core`, `--threads`, `--device`) → Task 9 ✓
- Running-on-a-VM scripts → Task 10 ✓
- Dataset provisioning (DVC + rsync) → Task 11 ✓
- §6/§7 API path → explicitly Phase 2, not in this plan ✓

**Type consistency:** stage names `pad/yolo/tubes/crop/vit/trigger` identical across `stage_timer` callers, `run_core.STAGES`, and `report.STAGES`. `BenchSequence` fields (`key/label/frame_count/frames`) consistent between `dataset.py` and `run_core.py`. `machine_info(*, device=...)` signature matches all call sites in `cli.py`. `write_report`/`summarize`/`ResourceSampler.timeline()` signatures match `cli.py` usage.

**Placeholder scan:** none — every step has concrete code/commands and expected output.
