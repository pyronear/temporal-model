# Benchmark Phase 2 — API end-to-end design

**Date:** 2026-06-09
**Package:** `benchmark/` (`temporal-model-benchmark`) + `api/` instrumentation
**Status:** approved, pending implementation plan
**Supersedes:** §6–§7 ("Phase 2") of `2026-06-09-benchmark-package-design.md`,
which were written before the detection-cache / eval-only-trigger / S3-flow
changes landed on `main`.

## Goal

Measure realistic **production serving latency** for the temporal smoke
classifier on the CPU VM: the full request path — HTTP → S3 frame fetch →
cached detection → tube building → classifier → JSON serialization — with a
**server-side per-stage breakdown**, under two cache regimes (cold worst-case
and warm steady-state).

Phase 1 measured the in-process `predict()` compute path. Its headline numbers
now **overstate production** because the architecture moved on:

- **`trigger_search` is eval-only** (`predict(..., compute_trigger=False)` in
  serving), so the 34% it consumed in the Phase 1 CPU run is **not** in the
  served path.
- **Detection is cached** (`DetectionCache`, `TEMPORAL_API_DETECTION_CACHE_SIZE`
  default 4096). In an ongoing event the model is called on a growing window, so
  the expensive `detector` stage is almost fully amortized after the first call.
- **S3 fetch and detection orchestration** live in the serving layer
  (`app.py` and `ModelRunner`), outside `core.predict`.

Phase 2 measures that real path.

## Context — the current serving flow

`POST /predict` (`api/src/temporal_model/api/app.py`):

1. `fetch_frames(s3_client, bucket, body.frames, tmp_dir)` — blocking boto3 S3
   download to a temp dir, run in a threadpool. **(`s3_fetch`)**
2. `runner.predict(paths)` → `ModelRunner._predict_sync` (under an asyncio lock,
   in a worker thread):
   - `load_sequence(paths)` → `Frame`s,
   - resolve detections against the `DetectionCache`: hits served from cache,
     **misses** run through `model.detect()` (the YOLO forward) and cached,
     **(`detector`)**
   - `model.predict(frames, frame_detections=resolved)` → `pad`, `tubes`,
     `crop`, `classifier` stages (`detector` block is a ~0 ms re-resolve since
     all detections are supplied; `trigger_search` off).
3. `to_response(out, ..., verbose=...)` — serialize.

The serving path is therefore **6 timed stages**: `s3_fetch`, `detector`,
`pad`, `tubes`, `crop`, `classifier`. Production is **single-stream** (the
runner serializes inference behind an `asyncio.Lock`).

## Design

### 1. Server-side profiling (`api` package)

New setting **`TEMPORAL_API_PROFILE: bool = False`** (env
`TEMPORAL_API_PROFILE`). Default off ⇒ no `StageTimer`, no syncs, no schema
change visible — current behavior bit-for-bit unchanged.

When on, a **request-scoped profiling collector** spans the whole request:

- `app.predict` creates a `StageTimer` (the Phase 1 `core.stage_timer.StageTimer`)
  and a small `profile` dict, times the S3 fetch under `timer.stage("s3_fetch")`,
  then calls `runner.predict(paths, timer=timer, profile=profile)`.
- `ModelRunner._predict_sync` times its real detection work under
  `timer.stage("detector")`, writes `n_frames` / `cache_hits` / `cache_misses`
  into `profile`, and threads the **same** `timer` into
  `model.predict(frames, frame_detections=resolved, timer=timer)` so `pad` /
  `tubes` / `crop` / `classifier` are recorded too. (The `predict()` internal
  `detector` block adds ~0 ms — re-resolution only — and harmlessly accumulates
  onto the same `detector` key.)
- Result `profiling = {**timer.as_dict(), "total_ms": ..., **cache_counts}`.

Surfaced two ways:

- **Logged** as one structured JSON line per request (always, when profiling
  on) — so a server under load is observable without `verbose`.
- **Added to the `?verbose=true` response** under `details.profiling` — one new
  **optional** field on the `Details` schema (`api/.../schemas.py`), populated
  only when profiling is on and verbose is requested.

Threading: a `timer`/`profile` kwarg is added to `ModelRunner.predict` /
`_predict_sync` (defaults `None` → no-op). `core.predict` already accepts
`timer`. This is the one cross-cutting change to `api`.

### 2. Benchmark client (`benchmark/run_api.py`)

`run_api(store, base_url, *, passes, warmup, limit, timestamp) -> pd.DataFrame`.
Runs **on the VM against `http://localhost:8000`** (MinIO is local to the VM, so
this isolates serving cost from internet latency). Single-stream (sequential
requests).

Two passes against the same server session:

- **Cold** — POST each sequence's **full** frame-key list once. Frames are
  distinct across sequences, so cache hits ≈ 0 ⇒ worst-case "first alert"
  latency (every frame detected).
- **Warm** — for each sequence, POST **growing prefixes** (keys `1..k` for
  `k = infer_min .. N`). Within a sequence, follow-up calls hit the cache for
  earlier frames ⇒ steady-state per-call cost (detector mostly amortized), as
  in an ongoing event.

Each request → one raw row: `pass` (`cold`/`warm`), `key`, `prefix_len`,
`e2e_ms` (client wall-clock around the HTTP call), `http_status`, and — when
the server returns it (`?verbose=true` + profiling on) — the `profiling` stage
columns (`s3_fetch_ms`, `detector_ms`, `pad_ms`, `tubes_ms`, `crop_ms`,
`classifier_ms`, `total_ms`) plus `cache_hits` / `cache_misses` / `n_frames`.
`warmup` sequences are replayed first and discarded. Non-200s recorded per
request, not fatal.

Each sequence's S3 keys come from the dataset loader (Phase 1 `dataset.py`);
the key for a frame is its store-relative path (the same key used at upload).

### 3. Reporting (extend `benchmark/report.py`)

A new `summarize_api(df)` + `write_api_report(...)` that aggregate **cold and
warm separately**:

- e2e latency p50/p90/p99/mean per pass,
- mean server stage breakdown per pass (incl. `s3_fetch`), and server-vs-e2e
  gap (HTTP/framework overhead = `e2e_ms - profiling.total_ms`),
- warm-pass **cache hit rate** and the detector-amortization effect
  (cold `detector_ms` vs warm `detector_ms`),
- throughput per pass.

Reuses Phase 1 `machine.py` (CPU/RAM metadata) and `resources.py` (sampler),
and writes the same self-describing layout to
`data/08_reporting/<host>-api-<timestamp>/` (raw.parquet, resources.parquet,
summary.json, plots, report.md). Plots: cold-vs-warm e2e distribution,
stage-breakdown bars per pass, warm per-call latency vs prefix length.

### 4. CLI (`benchmark/cli.py`)

Add an **`api`** subcommand:

```
temporal-benchmark api --url http://localhost:8000 \
    --store data/03_primary/sequences \
    [--passes cold,warm] [--warmup 3] [--limit N] [--timestamp ...] [--out ...]
```

### 5. Provisioning (`benchmark/scripts/`)

- **`provision_api_vm.sh <host>`** — install Docker on the VM, then bring up the
  `api/docker-compose.yml` stack (API + MinIO + `createbuckets`) with
  `TEMPORAL_API_PROFILE=1` (and `TEMPORAL_API_DEVICE` left as CPU). The compose
  stack already ships MinIO (`frames` bucket, `minioadmin` creds,
  `http://minio:9000`). `api/models/model.zip` is already on the VM from
  Phase 1 (or `make fetch-model`).
- **`upload_frames_to_minio.py`** — using boto3 against the VM's MinIO
  (`http://localhost:9000`, `minioadmin`), upload every pyro-annotator frame to
  the `frames` bucket under its store-relative key — the exact keys
  `run_api.py` POSTs. Idempotent (skip existing).

### Error handling / edge cases

- **Profiling off (default):** no timer, no `profile` dict, no `details.profiling`
  field — zero overhead, unchanged responses. Covered by existing API tests.
- **`verbose=false` with profiling on:** timings are logged but not in the
  response body; `run_api` always sends `?verbose=true`.
- **HTTP non-200 / inference error:** recorded with `http_status`; pass
  continues.
- **Warm pass on a sequence shorter than `infer_min`:** only the full-length
  call is issued (no sub-`infer_min` prefixes).
- **Cache bleed between cold and warm:** cold runs first; warm's growing windows
  re-detect the same frames cold would have, so warm hit rate is measured
  per-request from the server counts, not assumed.

### Testing

- `api`: unit test that `TEMPORAL_API_PROFILE=1` adds `details.profiling` with
  the six stage keys + cache counts on a verbose response, and that it is
  **absent** when the flag is off (model mocked, S3 via moto — matches existing
  `api/tests` conventions). Existing API tests must pass unchanged with the flag
  default-off.
- `benchmark`: `summarize_api` aggregation test (known rows → known cold/warm
  percentiles, hit rate, server-vs-e2e gap). Heavy HTTP/Docker runs are not
  unit-tested (repo convention).

### Out of scope (YAGNI)

Concurrency / load testing (production is single-stream behind the lock), real
OVH S3 (MinIO only), GPU, driving the client from a remote machine (run on the
VM against localhost).

## Files

**Edited (`api`):**
- `settings.py` — add `profile: bool = False`.
- `app.py` — request-scoped `StageTimer`, time `s3_fetch`, thread timer/profile.
- `model_runner.py` — `predict(..., timer=None, profile=None)`; time `detector`,
  record cache counts, thread timer into `model.predict`.
- `schemas.py` — optional `profiling` field on `Details`; populate in
  `to_response`.

**New (`benchmark`):**
- `src/temporal_model/benchmark/run_api.py`
- `report.py` — add `summarize_api` / `write_api_report`.
- `cli.py` — add `api` subcommand.
- `scripts/provision_api_vm.sh`, `scripts/upload_frames_to_minio.py`
- `tests/test_report_api.py`

**Reused unchanged:** `dataset.py`, `machine.py`, `resources.py`,
`core/stage_timer.py`.
