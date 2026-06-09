# API detection cache: stop re-detecting frames a growing sequence has already seen

**Date:** 2026-06-09
**Status:** Approved (brainstorm)
**Scope:** `api` (detection cache, orchestration, cache observability) + a small
pure detection-injection seam in `core`. No model behavior change.

## Goal

Cut the redundant compute the temporal-model API does when a client re-calls
`POST /predict` on a sequence that grows by one frame every 30 s. Today every
call re-runs YOLO over the **entire** sequence; we want each frame detected
**once** and its detections reused on subsequent calls — with no change to the
verdict — so the API comfortably absorbs production load. (Cheaper calls also
reduce HTTP connection-timeout risk, but explicit timeout handling is out of
scope here.) See **Numerical determinism** below for the precise equivalence
guarantee.

## Background / current state

The pipeline is stateless. `POST /predict` (`api/app.py`) fetches frames from S3
and calls `runner.predict(paths)` (`api/model_runner.py`), which serializes
inference behind one `asyncio.Lock` and runs the model in a worker thread.

The model follows a template-method contract (`core/protocol.py`):

```
predict_sequence(paths) → load_sequence(paths) -> list[Frame] → predict(frames)
```

`predict(frames, *, timer=None, compute_trigger=False)` runs the stages
(`core/model.py`): `pad → detector → tubes → crop → classifier → trigger_search`.
Two facts from the recent refactor shape this design:

- **`trigger_search` is eval-only now.** Prod calls with `compute_trigger=False`,
  so the hot path is `detector → tubes → crop → classifier` plus a cheap inline
  decision (see `docs/specs/2026-06-09-trigger-search-eval-only-design.md`).
- The **detector stage dominates**. It is `run_yolo_on_frames(yolo, truncated,
  …) -> list[FrameDetections]` (`core/inference.py`), a single batched YOLO call
  over all frames. In the common **no-smoke** case there are no kept tubes, so
  `crop`/`classifier` are skipped and the detector is essentially the *entire*
  per-call cost.

`FrameDetections` (`core/types.py`, public via `core/__init__.py`) is
`{frame_idx, frame_id, timestamp, detections: list[Detection]}` — pure data, no
tensors, a handful of floats per box. `Frame.frame_id` is the filename stem,
a **stable per-frame identity** carrying site + timestamp.

`protocol.TemporalModel.load_sequence`'s docstring already names
*"attach cached YOLO detections"* as an intended override point — the seam below
lands on that intent.

## Problem

1. **O(N²) regrowth per camera.** A sequence that grows to N frames is detected
   1 + 2 + … + N times across N calls. N is capped short (~10–30) but the waste
   is pure repetition of deterministic work.
2. **Correlated bursts.** A real plume trips up to **3 nearby cameras at once**,
   all on the expensive (has-tubes) path, all serialized behind the one lock —
   the worst moment to be slow. Cheaper per-call work drains that queue faster.

## Non-goals (explicitly out of scope)

Ruled out during brainstorming as unnecessary at this scale (≤3 correlated
cameras, single process, short sequences):

- **Per-camera positive-latch** (freeze verdict after first positive) — gated on
  unknown client post-positive behavior; not needed for capacity. Revisit only
  if load grows and we confirm client semantics.
- **Classifier / tube prefix reuse** — adding frames can retroactively change
  tube composition (merges, gap interpolation); unsafe and not worth it at N≤30.
- **Worker pool / removing the lock** — on CPU one prediction already uses all
  cores; pooling would just time-slice. GPU inference is not reentrant.
- **Backpressure / load-shedding (429/503 on volume)** — wrong for fire
  detection; a burst *is* the event you must serve.
- **Async job pattern (202 + poll/webhook)** — changes the synchronous contract
  the black-box client depends on; over-engineering here.
- **S3 fetch timeout safety** (bounded boto3 connect/read timeouts + retries) —
  deferred for now; not part of this cache work.
- **Whole-response / per-sequence cache** — clients send a longer sequence each
  call, so it rarely hits.
- **Putting the cache in `core`** — it is a serving optimization; in `core` it
  would make the model stateful and corrupt `benchmark`/`eval` timings.

## Design

State lives in the API; `core` stays pure and deterministic.

### 1. `core` seam — pure detection injection (no state)

Add to `BboxTubeTemporalModel`:

- **`detect(frames: list[Frame]) -> list[FrameDetections]`** — a thin public
  wrapper exposing the existing detector stage (`run_yolo_on_frames` with the
  model's configured `confidence_threshold`/`iou_nms`/`image_size`/device). Pure.
- **`predict(frames, *, frame_detections: dict[str, FrameDetections] | None =
  None, timer=None, compute_trigger=False)`** — in the `detector` stage:
  - if `frame_detections is None`: behave exactly as today (full
    `run_yolo_on_frames`);
  - otherwise: for each (post-pad `truncated`) frame, reuse
    `frame_detections[frame.frame_id]` when present, collect the misses, run
    `run_yolo_on_frames` on the **miss subset only**, and merge — preserving the
    truncated frame order. Each reused `FrameDetections` is **re-stamped** with
    its current positional `frame_idx` (the cached `frame_idx` is from a prior
    call and must not leak).

`predict_sequence` and the `TemporalModel` protocol are **not** threaded with the
new parameter (YAGNI — only the API orchestrates injection, and it holds a
concrete `BboxTubeTemporalModel`). `benchmark`/`eval` keep calling the
no-injection path, so their detector timings stay honest. Default-path output is
bit-for-bit unchanged.

### 2. API — detection cache in `ModelRunner`

`ModelRunner` gains an LRU keyed by `frame_id`, valued by `FrameDetections`:

- **Capacity:** default **4096** entries (~4–8 MB; negligible vs the model).
  Configurable via `TEMPORAL_API_DETECTION_CACHE_SIZE` (`0` disables the cache →
  always full detection, for parity/debug).
- **Implementation:** a small size-bounded LRU (`collections.OrderedDict` or
  `cachetools.LRUCache`).
- **Lifetime / invalidation:** the cache lives on the `ModelRunner` instance,
  which is created at load and replaced on model reload — so a new model wipes
  the cache for free. Detection config is fixed per loaded model, so it is **not**
  part of the key.

`runner.predict` orchestration (entirely under the existing inference lock, so
cache reads/writes and the YOLO call are serialized and thread-safe):

```python
frames = self._model.load_sequence(paths)
misses = [f for f in frames if f.frame_id not in self._cache]
for fd in self._model.detect(misses):        # YOLO on misses only (often 1)
    self._cache[fd.frame_id] = fd
dets = {f.frame_id: self._cache[f.frame_id] for f in frames}
out = self._model.predict(frames, frame_detections=dets)
```

Note: padding duplicates frames but they **share a `frame_id`**, so duplicates
collapse to one cache entry and one detection — a small bonus.

### 3. API — cache observability

Log per-call: detection-cache hits/misses, sequence length, and total latency.
This confirms the cache is working (hit rate climbs as a sequence grows) and
surfaces how per-call cost scales — enough to validate the feature without any
additional metrics infrastructure.

## Data flow

```
POST /predict ─▶ fetch_frames (outside lock)
              ─▶ runner.predict (under lock):
                   load_sequence → cache lookup → detect(misses) → cache update
                   → predict(frames, frame_detections=…) → decision
              ─▶ reshape DTO ─▶ 200
```

## Edge cases & correctness

- **Positional `frame_idx`:** reused entries re-stamped to current position.
- **Partial hits:** miss subset detected; merge preserves order.
- **Padding:** duplicate frames share `frame_id` → deduped in cache.
- **Cache disabled (`size=0`):** bit-for-bit identical to today (each call
  detects the whole sequence in one batch, exactly as before).
- **Thread-safety:** all cache access is under the inference `asyncio.Lock`.

## Numerical determinism

The cache reuses the exact `FrameDetections` it stored, so within a single call
`predict(frames, frame_detections=detect(frames))` is **bit-for-bit identical**
to `predict(frames)` (same YOLO batch, pinned by a parity test with a
deterministic stub detector).

Across calls it is **not** bit-for-bit identical to a single-shot run of the
whole sequence, and that is expected: YOLO is **not batch-size-invariant** (CPU
measurement: detections of the same frame differ by ~`7e-7` between a batch of 8
and frame-by-frame batches; run-to-run at a fixed batch is exactly `0`). A warm
cache detected each frame in a small batch when it first arrived, whereas a
cold full-sequence run detects them all in one batch — so the calibrated
`probability` can differ by ~`1e-9`. The **binary verdict (`is_smoke`) is
unaffected.**

This is a property of the growing-sequence deployment, **not** of the cache:
even without the cache, the client sends a longer sequence each tick, so the
detector already runs at a different batch size every call. The cache changes
*which* batch grouping is used, not whether it varies. The guarantee we provide
is therefore: **identical verdict; probability equal up to YOLO's inherent
batch-size float nondeterminism (~`1e-9`).**

## Testing

**core**
- `detect(frames)` returns the same `FrameDetections` as `run_yolo_on_frames`.
- Parity: `predict(frames, frame_detections=detect(frames))` ==
  `predict(frames)` (bit-for-bit `is_positive`, `probability`, details).
- Miss-only detection: with a mock YOLO, assert it is called with exactly the
  uncached subset, and reused entries are re-stamped to correct `frame_idx`.

**api**
- Cache hit avoids re-detection: second call on a grown sequence invokes
  `model.detect` only with the new frame(s).
- Response is identical with the cache disabled (`size=0`) vs the pre-change
  path.
- LRU eviction at capacity; cache reset on model reload.

## Configuration summary (new `TEMPORAL_API_*`)

| Var | Default | Meaning |
|---|---|---|
| `DETECTION_CACHE_SIZE` | `4096` | LRU capacity; `0` disables |

## Acceptance criteria

- A grown-by-one re-call detects only the new frame(s); detector cost per call
  in the no-smoke case is ~constant rather than O(N).
- With the cache **disabled** (`size=0`), responses are byte-identical to the
  pre-change behavior.
- With the cache **enabled**, the verdict (`is_smoke`) matches the cache-off
  run and `probability` matches up to YOLO's batch-size nondeterminism
  (~`1e-9`) — see Numerical determinism.
- `core` default path (`predict(frames)`, no injection) is unchanged;
  `benchmark`/`eval` timings unaffected.
- Per-call cache-hit + latency logging is emitted.
