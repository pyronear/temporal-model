# temporal-model-benchmark

Measures the temporal smoke classifier's **latency, throughput, and resource
usage** on a given machine, with a **per-stage breakdown** of where inference
time goes. Built to run on the different VMs we might use in production (starting
CPU-only) so their numbers are directly comparable.

Import as `temporal_model.benchmark`. Depends on `temporal-model-core`. See the
design spec: `docs/specs/2026-06-09-benchmark-package-design.md`.

> **Scope.** Phase 1 (this package today) benchmarks the **core in-process
> path** — calling `BboxTubeTemporalModel.predict()` directly. The API
> end-to-end path (`run_api.py`, `TEMPORAL_API_PROFILE`) is Phase 2 and not yet
> implemented.

## What it measures

`predict()` runs a six-stage pipeline. The benchmark times each stage
separately, so a single run tells you *which* stage dominates — not just the
total latency.

| stage | what happens | typically heavy on |
|---------|--------------------------------------------------|--------------------|
| `pad` | truncate to `max_frames`, pad short sequences | nothing (µs) |
| `yolo` | one batched YOLO detection call over all frames | GPU / CPU |
| `tubes` | greedy IoU tube building + filtering | nothing (µs) |
| `crop` | open each frame from disk, crop/resize patches | disk IO + CPU |
| `vit` | one batched ViT (DINOv2) classifier forward | GPU / CPU |
| `trigger` | re-score tube prefixes to find first-crossing | GPU / CPU |

The split shifts a lot with hardware: on a GPU, `yolo` usually dominates; on a
CPU VM, `vit` and `trigger` grow relative to it. Surfacing that flip is the
whole point.

### How the timing works

A small `StageTimer` lives in `core` (`temporal_model.core.stage_timer`). It is
**opt-in**: `predict(frames, timer=StageTimer(device))` records per-stage
wall-clock ms; `predict(frames)` with no timer is bit-for-bit identical to the
unprofiled path (no timing, no CUDA syncs — the core parity tests still pass).
On CUDA the timer synchronises at each stage boundary so GPU stage times are
real, not just kernel-launch latency.

## Data layout (Kedro layers)

Data follows this repo's Kedro-style numbered layers (same as `train`/`eval`):

```
data/
├── 03_primary/sequences/pyro-annotator/   # input store — DVC-tracked
└── 08_reporting/<host>-<timestamp>/        # one self-describing dir per run
```

The input is the **pyro-annotator sequence store** (332 sequences, ~562 MB)
imported from the temporal-model-explorer. Each sequence is a directory with a
`meta.json` (ordered frame list + ground-truth label) plus frame images. Frame
counts span 3→30 and labels are mixed smoke/fp — the variation that drives the
latency spread.

`08_reporting` is gitignored (per-VM outputs). The input store is DVC-tracked;
the repo's blanket `**/data/**` ignore means the small `.dvc` pointer is
committed via `git add -f`.

## Install & get the data

```bash
make install          # uv sync — installs temporal-model-core editable + deps
dvc pull              # fetch the pyro-annotator store into data/03_primary/
```

You also need a packaged model. From the repo root:

```bash
make fetch-model      # downloads api/models/model.zip from HuggingFace (no creds)
```

## Run

```bash
temporal-benchmark core --model ../api/models/model.zip
```

This loads the model once, warms up, then times `predict()` over every sequence
in the store and writes a results dir under `data/08_reporting/`.

### Options

| flag | default | meaning |
|---------------------|----------------------------|--------------------------------------|
| `--store` | `data/03_primary/sequences`| sequence store root (recursive) |
| `--model` | *(required)* | path to `model.zip` |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--reps` | `5` | timed repetitions per sequence |
| `--warmup` | `3` | warmup sequences (discarded) |
| `--limit` | *(all)* | cap number of sequences (quick runs) |
| `--threads` | all CPU cores | `torch.set_num_threads()` (`os.cpu_count()`) |
| `--sample-interval` | `0.1` | resource sampler period (s) |
| `--out` | `data/08_reporting` | output parent dir |
| `--timestamp` | `run` | label for the results dir |

Example quick smoke run:

```bash
temporal-benchmark core --model ../api/models/model.zip \
    --reps 2 --warmup 1 --limit 5 --timestamp smoke
```

## Outputs

Each run writes `data/08_reporting/<host>-<timestamp>/`:

| file | contents |
|----------------------|--------------------------------------------------------|
| `raw.parquet` | one row per (sequence, rep): per-stage ms, total, frame count, #tubes, label, failed flag |
| `resources.parquet` | CPU/RAM (and GPU when present) utilisation timeline |
| `summary.json` | latency p50/p90/p99, throughput, mean stage shares, **+ full machine metadata** |
| `plots/*.png` | latency distribution, stage breakdown, latency-vs-frame-count, resource timeline |
| `report.md` | human-readable summary stitching all of the above together |

The machine metadata stamped into `summary.json`/`report.md` (host, CPU, GPU,
RAM, torch/cuda/python versions, thread count, device) is what makes runs from
different VMs comparable — read `report.md` first.

## Running on a VM

The target VMs are provisioned natively with `uv` (no Docker/dvc/S3 creds
needed). Three helper scripts in `scripts/` wrap the flow:

```bash
# from your machine (which has the data via `dvc pull`)
scripts/provision_vm.sh  ubuntu@<host>   # install uv, clone, python 3.12, deps, model
scripts/push_data.sh     ubuntu@<host>   # rsync data/03_primary store onto the VM

# on the VM
cd temporal-model && uv run temporal-benchmark core --model api/models/model.zip

# back on your machine
scripts/pull_results.sh  ubuntu@<host>   # rsync data/08_reporting back
```

The data travels by **rsync**, not dvc: VMs aren't assumed to have S3
credentials. DVC remains the in-repo source of truth; rsync is just the
transport. The same scripts work for any VM — only the host changes.

## Module map

| module | responsibility |
|----------------|--------------------------------------------------------|
| `dataset.py` | read the `meta.json` store → `BenchSequence` (core `Frame`s) |
| `machine.py` | capture host/CPU/GPU/torch metadata |
| `resources.py` | background CPU/RAM/GPU sampler (context manager) |
| `run_core.py` | drive `predict()` with a `StageTimer`, collect raw rows |
| `report.py` | aggregate raw rows → summary.json + plots + report.md |
| `cli.py` | `temporal-benchmark core` entry point |

## Develop

```bash
make test     # pytest (loader, machine, resources, report aggregation)
make lint     # ruff check
make format   # ruff format
```
