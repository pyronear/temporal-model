# Benchmark package — design

**Date:** 2026-06-09
**Package:** `benchmark/` (`temporal-model-benchmark`, import `temporal_model.benchmark`)
**Status:** approved, pending implementation plan

## Goal

Measure the temporal smoke classifier's **latency, throughput, and resource
usage** on the VMs we might run it on in production, and produce a clear
**breakdown of where inference time goes**. The same benchmark is run on
different VMs (starting with `ssh ubuntu@141.94.173.1`, an OVH host) so we can
compare hardware and spot bottlenecks before committing to a production target.

Two complementary views:

1. **Core in-process breakdown** — call `BboxTubeTemporalModel.predict()`
   directly and time each of its six pipeline stages. The only way to see the
   per-stage split (YOLO vs crop+IO vs ViT vs prefix-trigger).
2. **API end-to-end** — drive the running FastAPI `/predict` over HTTP for the
   realistic production latency (S3 frame fetch + threadpool + inference + JSON
   serialization), with the same per-stage breakdown surfaced from the server
   when profiling is enabled.

## Context

### The inference pipeline (what we are timing)

`BboxTubeTemporalModel.predict(frames)` (`core/src/temporal_model/core/model.py`)
runs these stages in order:

1. **pad/truncate** — cheap list ops (`pad_frames_*`, truncate to `max_frames`).
2. **YOLO detection** — `run_yolo_on_frames`, one batched ultralytics call over
   all frames. GPU/CPU heavy.
3. **tube building** — `build_tubes` + `build_tubes_for_inference`, pure CPU,
   cheap.
4. **patch crop + IO** — `crop_tube_patches` per kept tube: opens each frame
   image from disk (`PIL.Image.open`), crops/resizes/normalizes. Includes disk
   IO; CPU.
5. **ViT scoring** — `score_tubes`, one batched classifier (DINOv2 ViT +
   transformer head) forward over all tubes. GPU/CPU heavy.
6. **first-crossing trigger** — `find_first_crossing_trigger`, re-runs the
   classifier serially on tube *prefixes* to find the earliest firing frame.
   Can be the dominant cost when tubes are long.

The API serves one sequence per `POST /predict` request and **serializes
inference behind an asyncio lock** (`api/src/temporal_model/api/model_runner.py`),
so production is effectively single-stream. Throughput is therefore reported as
sequential sequences/sec and frames/sec, not concurrent load.

### The dataset

The realistic dataset is the **sequence store** from
`vision-rd/experiments/temporal-models/temporal-model-explorer`
(`data/03_primary/sequences/`). We use the **`pyro-annotator` source only**:

- **332 sequences, ~562 MB**, nested `pyro-annotator/<org>/<camera>/<seq>/`.
- Each sequence dir has a `meta.json` (a `SequenceMeta`: ordered `FrameRef`s +
  ground-truth `label`) plus the frame image files.
- **Frame counts span 3 → 30** (well distributed; a spike of 41 sequences at 30
  suggests a cap). Frame count is the main latency driver — good coverage.
- **Labels: 273 fp · 44 smoke · 15 unknown.** Smoke sequences exercise the full
  tube → ViT → trigger path; fp sequences exercise the cheap/no-tube path.

The store's `meta.json` format (from explorer `store.py`):

```json
{
  "key": "...", "sequence_id": "...", "source": "...",
  "label": "smoke|fp|unknown", "label_detail": null, "label_source": "...",
  "frames": [{"file": "images/detection_5.jpg", "detection_id": 5, "created_at": "..."}],
  "camera_id": null, "camera_name": "...", "organization_id": null,
  "organization_name": "...", "started_at": "..."
}
```

`frames` order is the time axis. We will **re-read this format** with a small
self-contained loader (no dependency on the explorer or `pyrocore`), emitting
`temporal_model.core.protocol.Frame(frame_id, image_path, timestamp)`.

### Repo conventions this package follows

Each package is `temporal-model-<name>`, layout `src/temporal_model/<name>/`,
hatchling build, depends on `core` via
`temporal-model-core = { path = "../core", editable = true }`, has its own
per-package `.dvc` (remote `s3://pyro-vision-rd/dvc/temporal-model/<pkg>/`),
`Makefile`, and ruff config. `benchmark/` mirrors `eval/`.

## Design

### Architecture overview

```
benchmark CLI ──┬── run_core ──> predict(frames, timer) ──> raw rows
                └── run_api  ──> POST /predict?verbose ──> raw rows
                                     (+ resources sampler running alongside)
                raw rows ──> report ──> raw.parquet + summary.json + plots + report.md
                machine metadata stamped on every run
```

### 1. Core change — shared stage timer (`core`)

The single source of truth for the per-stage breakdown. New module
`core/src/temporal_model/core/stage_timer.py`:

```python
class StageTimer:
    """Accumulates per-stage wall-clock ms. CUDA-sync aware for honest GPU times."""
    def __init__(self, device: torch.device | None = None) -> None: ...
    @contextmanager
    def stage(self, name: str): ...      # syncs CUDA at enter/exit when device is cuda
    def as_dict(self) -> dict[str, float]: ...  # {"yolo": 12.3, "crop": 4.1, ...}
```

`predict()` (and `predict_sequence`) gain an **optional keyword** `timer:
StageTimer | None = None`. Each of the six stages is wrapped in
`with timer.stage("<name>"):` **only when a timer is provided**. When
`timer is None` the calls are skipped entirely (a module-level no-op context
manager) — control flow and numerics are **bit-for-bit identical to today**, so
`test_model_parity.py` and the rest of the suite keep passing without changes.

CUDA timing: when the model device is CUDA and a timer is active, the timer
calls `torch.cuda.synchronize()` at each stage boundary so GPU stage times are
real (not just kernel-launch times). These syncs run **only when profiling is
on**.

Stage names: `pad`, `yolo`, `tubes`, `crop`, `vit`, `trigger`.

> This is the only change outside the new package. `predict()`'s signature in
> the `TemporalModel` protocol gains the optional keyword; all existing callers
> are unaffected.

### 2. Dataset loader (`benchmark/dataset.py`)

Self-contained reader of the `meta.json` sequence store.

```python
@dataclass
class BenchSequence:
    key: str
    label: str            # smoke | fp | unknown
    frame_count: int
    frames: list[Frame]   # temporal_model.core.protocol.Frame

def iter_sequences(store_dir: Path) -> Iterator[BenchSequence]:
    """Yield one BenchSequence per meta.json under store_dir (recursive)."""
```

No `pyrocore` / explorer import — reads the JSON, builds `core` `Frame`s with
`image_path = seq_dir / ref.file`. Mirrors explorer `store.build_frames`.

### 3. Machine metadata (`benchmark/machine.py`)

```python
def machine_info() -> dict:
    """hostname, platform, cpu_model, cpu_count_physical/logical, ram_total_gb,
       gpu_name (or None), torch_version, cuda_version, python_version,
       device_used, model_version (from model.zip manifest)."""
```

Stamped into `summary.json` and `report.md` so every run is self-describing and
runs from different VMs are directly comparable.

### 4. Resource sampler (`benchmark/resources.py`)

A background thread sampling at a fixed interval (default 100 ms) while a run is
in progress:

- **CPU% + RAM** via `psutil` (process + system).
- **GPU util + mem** via `pynvml` (NVML). Fallback to polling `nvidia-smi`;
  if neither is available, GPU metrics are omitted (CPU-only VMs are fine).

```python
class ResourceSampler:
    def __enter__(self): ...   # starts the thread
    def __exit__(self, ...): ...  # stops it
    def timeline(self) -> list[dict]: ...   # [{t, cpu_pct, ram_gb, gpu_util, gpu_mem_gb}]
    def peaks(self) -> dict: ...
```

Graceful degradation: missing libs/hardware reduce the metric set, never crash.

### 5. Core-path runner (`benchmark/run_core.py`)

```python
def run_core(store_dir, model_path, *, device, reps, warmup, limit) -> pd.DataFrame:
```

- Load the model once via `BboxTubeTemporalModel.from_package(model_path, device)`.
- `warmup` sequences are predicted and discarded first (CUDA/cuDNN init, timm /
  ultralytics first-call compilation).
- For each sequence (optionally capped by `limit`), run `reps` timed
  `predict(frames, timer=StageTimer(device))` calls.
- Emit **one raw row per (sequence, rep)**: `key, label, frame_count,
  n_kept_tubes, rep, total_ms`, one column per stage (`pad/yolo/tubes/crop/vit/
  trigger`_ms), plus a `failed` flag for sequences that raise (recorded, not
  fatal). `n_kept_tubes` comes from the prediction `details`.

The `ResourceSampler` wraps the whole timed section; its timeline + peaks are
saved alongside the raw table.

### 6. API-path runner (`benchmark/run_api.py`)

```python
def run_api(store_keys, base_url, *, reps, warmup, ...) -> pd.DataFrame:
```

- `POST {base_url}/predict?verbose=true` with the sequence's ordered S3 keys,
  measuring **client-side wall-clock e2e latency**.
- When the API has profiling enabled (below), read the server's per-stage
  block from `details.profiling` (which additionally includes the
  **`s3_fetch`** stage that lives in the API, outside `core.predict`).
- One raw row per (sequence, rep): `key, e2e_ms, http_status`, plus the server
  stage columns when present.
- Assumes the same pyro-annotator frames exist in the VM's S3/MinIO under the
  expected keys (provisioning that is an operational step, not part of this
  package). Non-200s are recorded per sequence, not fatal.

### 7. API instrumentation (env-var gated)

In `api`: new setting `TEMPORAL_API_PROFILE` (default off). When on,
`ModelRunner.predict` constructs a `StageTimer` and passes it into
`predict_sequence`; the resulting stage timings (including the API-level
`s3_fetch` stage) are:

- **logged** as one structured JSON line per request, and
- **added to the verbose response** under `details.profiling`
  (`{stage: ms}` + `total_ms`).

Schema: one **optional** `profiling` field added to the API/details response
schema (`api/.../schemas.py` and/or `core/.../details_schema.py`). Off by
default ⇒ no schema change visible, no overhead, current behavior unchanged.

### 8. Report / aggregation (`benchmark/report.py`)

From the raw table:

- **`summary.json`** — machine metadata + per-stage and total latency p50/p90/
  p99/mean, throughput (sequences/sec, frames/sec), mean stage-share %, counts
  (n sequences, n failed), resource peaks. Compact and diffable for cross-VM
  comparison.
- **Plots (PNG)** via matplotlib:
  - total-latency distribution (histogram / violin),
  - stage-breakdown stacked bar (mean ms per stage),
  - latency-vs-frame-count scatter,
  - throughput bar (seq/s, frames/s),
  - CPU/GPU-utilization timeline.
- **`report.md`** — stitches machine info + summary table + embedded plots into
  one README-style file telling the whole story for that VM.

### 9. CLI (`benchmark/cli.py`, script `temporal-benchmark`)

```
temporal-benchmark core --store <dir> --model <model.zip> [--device auto]
                        [--reps 5] [--warmup 3] [--limit N] [--out <dir>]
temporal-benchmark api  --url http://host:8000 --store <dir>
                        [--reps 5] [--warmup 3] [--limit N] [--out <dir>]
```

`--device auto` → cuda if available else cpu (matches explorer `run_models`).

### Output layout (self-describing per run)

```
benchmark/results/<host>-<timestamp>/
├── raw.parquet        # one row per (sequence, rep)
├── resources.parquet  # sampler timeline
├── summary.json       # machine meta + aggregates
├── plots/*.png
└── report.md
```

`<timestamp>` is supplied by the CLI at invocation; results dirs are not DVC
tracked (they are per-VM outputs, committed selectively or kept locally).

### Dataset provisioning (DVC)

Copy the `pyro-annotator` source into `benchmark/data/sequences/pyro-annotator/`,
`dvc add` it, and push to this package's remote
(`s3://pyro-vision-rd/dvc/temporal-model/benchmark/`). One `dvc pull` provisions
any VM. The store is the input to the **core** path; the **api** path
additionally needs the same frames in that VM's S3/MinIO (operational).

## Error handling / edge cases

- **No GPU / no NVML:** resource sampler emits CPU-only metrics; `machine_info`
  reports `gpu_name=None`; `--device auto` selects CPU. No crash.
- **A sequence raises in `predict`:** recorded with `failed=True` and skipped;
  the run continues and reports the failure count.
- **Empty / zero-tube sequence:** valid — stages 4–6 are cheap/empty; timed
  normally (this is the fp-path cost we want to measure).
- **API non-200:** recorded per sequence with `http_status`; not fatal.
- **`timer is None` (production / parity):** stage wrapping is skipped entirely;
  behavior identical to today.

## Testing

Unit tests (heavy model runs are not unit-tested, matching repo convention):

- `test_stage_timer.py` — no-op path leaves results unchanged; active timer
  records exactly the expected stage names; `as_dict` sums correctly.
- `test_dataset.py` — a fixture `meta.json` store loads into the right ordered
  `Frame`s, frame counts, and labels.
- `test_report.py` — a known raw table aggregates to known p50/p90/p99,
  throughput, and stage-share numbers.
- `test_machine.py` — `machine_info` returns the required keys and degrades when
  GPU libs are absent (monkeypatched).

A `core` parity guard already exists (`test_model_parity.py`); it must keep
passing unchanged, demonstrating the opt-in timer is non-invasive.

## Out of scope (YAGNI)

- Cross-VM overlay/compare tooling — each run is self-describing; diff
  `summary.json`s by hand first. Add later if it proves needed.
- Synthetic-data mode; the `sdis-77` / `sis-67` sources.
- API concurrency / load testing (production is single-stream behind a lock).
- DVC-tracking the per-run results.

## Files

**New package `benchmark/`:**
- `pyproject.toml`, `Makefile`, `.dvc/`, `README.md`
- `src/temporal_model/benchmark/`: `dataset.py`, `machine.py`, `resources.py`,
  `run_core.py`, `run_api.py`, `report.py`, `cli.py`, `__init__.py`
- `tests/`: `test_stage_timer.py` (or under core), `test_dataset.py`,
  `test_report.py`, `test_machine.py`
- `data/sequences/pyro-annotator/` (DVC-tracked)

**Edited:**
- `core/src/temporal_model/core/stage_timer.py` (new) + `model.py` (opt-in
  `timer` kwarg) + `protocol.py` (signature)
- `api/src/temporal_model/api/settings.py` (`TEMPORAL_API_PROFILE`),
  `model_runner.py` (build + pass timer, log), `schemas.py` /
  `core/details_schema.py` (optional `profiling` field)
- root `Makefile` (`PACKAGES := core train eval api benchmark`)
