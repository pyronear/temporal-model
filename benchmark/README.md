# temporal-model-benchmark

Measures the temporal smoke classifier's **latency, throughput, and resource
usage** on a given machine, with a **per-stage breakdown** of where inference
time goes. Built to run on the different VMs we might use in production (starting
CPU-only) so their numbers are directly comparable.

Import as `temporal_model.benchmark`. Depends on `temporal-model-core`. See the
design spec: `docs/specs/2026-06-09-benchmark-package-design.md`.

> **Scope.** Two benchmarks: the **core in-process path** (calling
> `BboxTubeTemporalModel.predict()` directly — `temporal-benchmark core`) and the
> **API end-to-end path** (HTTP → S3 → cached detection → serialization —
> `temporal-benchmark api`, see *API end-to-end benchmark* below).

## What it measures

### The model, in one breath

The temporal smoke classifier decides whether a short sequence of frames from
one camera contains smoke. End to end, `predict()` does:

```
frames ─▶ YOLO detects boxes ─▶ link boxes into temporal "tubes"
       ─▶ crop each tube's frames to 224×224 patches
       ─▶ score each tube with a ViT (DINOv2) classifier ─▶ decision + trigger frame
```

A **tube** is one candidate smoke plume tracked across frames (boxes in
consecutive frames linked by IoU). The model emits one logit per tube and, for a
positive, the **trigger frame** — the earliest frame at which it would have
fired (this is what time-to-detection is measured from).

### The six timed stages

The benchmark times each stage of `predict()` separately, so a single run tells
you *which* stage dominates — not just the total latency.

| stage | what it does | why it costs |
|-----------|----------------------------------------------------------------|-------------------------------|
| `pad` | truncate the sequence to `max_frames`; if too short, duplicate frames to a minimum length | pure Python list ops — microseconds |
| `detector` | one batched YOLO11 detection call over **all** frames at once, producing candidate boxes per frame | a CNN forward over full-resolution frames — heavy |
| `tubes` | link per-frame boxes into tubes (greedy IoU tracking), then filter/merge/interpolate them | small array bookkeeping — microseconds |
| `crop` | for every kept tube, open each frame **from disk** (PIL), crop around the box, resize to 224×224, normalize | image decode + disk IO + CPU resize |
| `classifier` | one batched forward of the tube classifier (ViT/DINOv2 backbone + transformer head) scoring all tubes' full-length patch stacks | a transformer forward — heavy |
| `trigger_search` | re-score growing **prefixes** of each qualifying tube to find the earliest firing frame (the trigger) — a serial loop of extra classifier forwards | repeated classifier calls — grows with tube length |

The split shifts a lot with hardware. On a GPU the single big `detector` forward
dominates (~86%) and the classifier side is cheap; on a CPU VM the picture flips
— `classifier` plus the serial `trigger_search` loop become ~40% of latency, because
each of those extra forwards is no longer nearly free. **Surfacing that flip is the
whole point**, and it points straight at what to optimize on a given machine (e.g.
the `trigger_search` loop's repeated forwards on CPU).

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

## API end-to-end benchmark

Measures the real serving path (HTTP → S3 fetch → cached detection → classifier
→ serialization) on the VM, with server-side per-stage timing via
`TEMPORAL_API_PROFILE`. The model's `trigger_search` stage is eval-only, so it
is not on the served path. Two passes:

- **cold** — each full sequence once: worst-case "first alert" latency (every
  frame detected).
- **warm** — growing prefixes per sequence: steady-state, with the detection
  cache amortizing the `detector` stage like an ongoing event.

```bash
# on the VM: bring up API + MinIO (profiling on) and upload frames
scripts/provision_api_vm.sh ubuntu@<host>
uv run python scripts/upload_frames_to_minio.py --store data/03_primary/sequences

# run the benchmark against the local API
uv run temporal-benchmark api --url http://localhost:8000 --store data/03_primary/sequences
```

Writes `data/08_reporting/<host>-api-<timestamp>/` (raw.parquet, summary.json,
report.md) with cold/warm e2e latency, per-stage breakdown (incl. `s3_fetch`),
and cache hit rate.

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
