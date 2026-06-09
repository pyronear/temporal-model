# Temporal Model API Service — Design

**Date:** 2026-06-02
**Status:** Approved
**Scope:** The `api` package's HTTP contract and serving design: load a packaged
`model.zip` and expose synchronous smoke-classification predictions over frames
referenced by S3 keys. Implementing the migrated `core` inference path is
out of scope (tracked separately); this spec defines the API surface and the
glue from request → `core` → response.

## Goal

Turn the scaffold `api` package into a runnable FastAPI service that:

1. Loads a packaged temporal smoke model (`model.zip`) once at startup.
2. Accepts an ordered sequence of frames (as S3 object keys) for one camera.
3. Runs the full bundled pipeline (YOLO → tubes → patches → temporal
   classifier → calibrator) via `BboxTubeTemporalModel.predict_sequence()`.
4. Returns a clean, reshaped JSON verdict, with an opt-in verbose breakdown.

## Background

The temporal model is a **bbox-tube temporal smoke classifier** originally built
in `vision-rd` (`lib/pyrocore`, `lib/bbox-tube-temporal`). Its packaged
deployment story is: *load `model.zip` and call `predict_sequence(frame_paths)`*.

Established facts from the `vision-rd` source that constrain this design:

- **`model.zip` bundles YOLO.** The package contains `manifest.yaml`,
  `yolo_weights.pt`, `classifier.ckpt`, `config.yaml`, and an optional
  `logistic_calibrator.json`. The detector runs *inside* the model, so callers
  send only frames — no detections. (Confirmed by the playground README: "The
  model runs YOLO itself, so only raw images are needed.")
- **Entry point:**
  ```python
  model = BboxTubeTemporalModel.from_package(zip_path, device=None)  # auto cuda→mps→cpu
  out   = model.predict_sequence(frame_paths)                        # list[Path], temporally ordered
  ```
- **Output type** (`pyrocore.TemporalModelOutput`):
  - `is_positive: bool`
  - `trigger_frame_index: int | None` (0-based; time-to-detection in *frames*;
    eval-only — always `None` on the serving path and omitted from the API DTO,
    see the response contract below)
  - `details: dict` (validated by `bbox_tube_temporal.details_schema.BboxTubeDetails`)
- **`details` schema** (`BboxTubeDetails`):
  - `preprocessing`: `num_frames_input`, `num_truncated`, `padded_frame_indices`
  - `tubes`: `num_candidates`, `kept: [KeptTube]`
    - `KeptTube`: `tube_id`, `start_frame`, `end_frame`, `logit`, `probability`
      (`None` if uncalibrated), `first_crossing_frame`, `entries: [TubeEntry]`
    - `TubeEntry`: `frame_idx`, `bbox: (cx,cy,w,h) normalized to [0,1] | None`,
      `is_gap`, `confidence`
  - `decision`: `aggregation` (`"max_logit" | "logistic"`), `threshold`,
    `trigger_tube_id`
- **Critical model contract:** the model parses **timestamps from frame
  filenames** (`<prefix>_<YYYY-MM-DDTHH-MM-SS>`) and relies on **temporal
  ordering**. The API must preserve each frame's filename (key basename) and the
  caller-provided order.

The model, training, and full `core` inference path will eventually live in this
repo. This spec covers only the API serving layer and its contract.

## Decisions

| Decision | Choice |
|---|---|
| Pipeline boundary | Full pipeline — client sends frames, API does detection→tubes→classify |
| Detector | Bundled in `model.zip`; API never receives detections |
| Frame source (v1) | S3 object keys only (no raw upload in v1) |
| Frame field | `frames: list[str]` — bare S3 keys, ordered |
| Key format | Bare keys; scheme prefixes (`s3://`, `http://`) rejected with `400` |
| Bucket / creds | Optional server-configured default bucket, overridable per request via `bucket`; creds via boto3 chain |
| S3 client | boto3 with configurable `endpoint_url` → AWS / OVH / MinIO unchanged |
| Ordering | Array order = temporal order; never re-sorted |
| Response style | Reshaped public DTO (not the raw `TemporalModelOutput`) |
| Verbosity | Lean default; `?verbose=true` adds a `details` block |
| Sync/async | Synchronous |
| Model metadata | `model: { name, version }` only (no architecture descriptor) |
| Model load | Once at startup via FastAPI `lifespan`; singleton |
| Concurrency | Inference serialized behind a lock, run in a threadpool |
| Spec location | `docs/specs/` |

## API Contract

### `POST /predict`

**Request** (`application/json`):

```jsonc
{
  "frames": [
    "cam12/2023-05-23/adf_site_999_2023-05-23T17-18-01.jpg",
    "cam12/2023-05-23/adf_site_999_2023-05-23T17-18-31.jpg",
    "cam12/2023-05-23/adf_site_999_2023-05-23T17-19-01.jpg"
  ],
  "bucket": "2eb7ac42fbbf-alert-api-2"
}
```

- `frames`: ordered list of **bare S3 object keys** (strings, ≥1), resolved
  against `bucket` (or the server-configured default when omitted).
- `bucket` (optional): S3 bucket to fetch the frames from. Falls back to
  `TEMPORAL_API_S3_BUCKET`; a request with neither is rejected with
  `400 invalid_request`. Added for alert-api stacks whose per-org bucket names
  are not known at deploy time.
- Array order **is** temporal order; the API never re-sorts.
- Each key's **basename** is the frame filename the model parses for timestamps
  (`<prefix>_<YYYY-MM-DDTHH-MM-SS>`).
- Entries containing a scheme (`s3://`, `http://`, …) are rejected with `400`.
- `list[str]` (not objects) by intent — YAGNI; widenable to `[{key, …}]` later
  without breaking simple callers.

**Default response** (`200`):

```jsonc
{
  "is_smoke": true,
  "probability": 0.98,
  "model": { "name": "bbox-tube-vit-dinov2", "version": "1.2.0" }
}
```

| Field | Source | Notes |
|---|---|---|
| `is_smoke` | `TemporalModelOutput.is_positive` | the verdict |
| `probability` | max kept-tube calibrated probability | see rule below; `null` only when uncalibrated |
| `model.name` | `manifest.variant` | architecture family |
| `model.version` | `manifest.model_version` | trained-model id; `null` for legacy packages without the field |

The API does **not** surface time-to-detection: `trigger_frame_index` and the
per-tube `first_crossing_frame` / `trigger_tube_id` are computed only on demand by
the core library (`predict(..., compute_trigger=True)`), used by offline eval, and
are intentionally omitted from the serving DTO. An explicit `?compute_trigger=true`
flag is tracked in #26 for when an HTTP consumer needs it.

**`probability` rule:** `probability` is `null` **iff the package is
uncalibrated** (no `logistic_calibrator`). For a calibrated model it is the
**highest** kept-tube probability (the strongest evidence), or `0.0` when no tubes
were kept — regardless of the `is_smoke` decision. On a positive sequence this is
the probability of whichever tube cleared the threshold; on a negative it is the
strongest sub-threshold evidence.

The "no tube found" outcome (`probability` `0.0`, `is_smoke` `false`) is *not*
separately flagged in the default response — it is distinguishable from a
tracked-but-low-scoring tube only via `?verbose=true` (`num_tube_candidates: 0`
and an empty `tubes` list). A default-level tube count was considered and
rejected (YAGNI): the lean default carries the verdict only, and consumers
needing the distinction request `details`.

**Verbose response** — `POST /predict?verbose=true` (`verbose` defaults to
`false`) returns the same top-level fields plus a `details` block:

```jsonc
{
  "is_smoke": true,
  "probability": 0.98,
  "model": { "name": "bbox-tube-vit-dinov2", "version": "1.2.0" },

  "details": {
    "decision": {
      "aggregation": "max_logit",
      "threshold": 0.5
    },
    "preprocessing": {
      "num_frames_input": 30,
      "num_truncated": 0,
      "padded_frame_indices": [],
      "num_tube_candidates": 5
    },
    "tubes": [
      {
        "tube_id": 7,
        "start_frame": 2,
        "end_frame": 12,
        "logit": 3.41,
        "probability": 0.98,
        "entries": [
          { "frame_idx": 2, "bbox": [0.693, 0.504, 0.0083, 0.0148], "is_gap": false, "confidence": 0.81 },
          { "frame_idx": 3, "bbox": null,                           "is_gap": true,  "confidence": null }
        ]
      }
    ]
  }
}
```

`details` maps from `BboxTubeDetails`, dropping the eval-only trigger fields
(`decision.trigger_tube_id` and per-tube `first_crossing_frame`), which are
always absent on the serving path (`compute_trigger=False`):
- `details.decision` ← `BboxTubeDetails.decision` (`aggregation`, `threshold`)
- `details.preprocessing` ← `BboxTubeDetails.preprocessing`, with
  `num_tube_candidates` ← `BboxTubeDetails.tubes.num_candidates`
- `details.tubes` ← `BboxTubeDetails.tubes.kept` (per-tube `probability` is
  `null` when uncalibrated; `bbox` is `(cx, cy, w, h)` normalized to `[0, 1]`
  (YOLO `xywhn` convention), or `null` on a gap frame)

### `GET /health`

```jsonc
{ "status": "ok", "model_loaded": true, "model_name": "bbox-tube-vit-dinov2", "model_version": "1.2.0" }
```

`model_loaded` is `false` (and `status` reflects not-ready) while the model is
still loading at startup.

### Errors

Error responses use a machine-readable body:

```jsonc
{ "detail": "human-readable message", "code": "snake_case_code" }
```

| HTTP | `code` | When |
|---|---|---|
| `400` | `invalid_request` | malformed body, empty `frames`, a key/bucket containing a scheme, an empty `bucket`, or no bucket available |
| `404` | `frame_not_found` | an S3 key does not exist in the bucket |
| `502` | `s3_unavailable` | S3 endpoint unreachable / fetch failure |
| `503` | `model_not_loaded` | request arrives before the model finishes loading |
| `500` | `inference_error` | unexpected failure inside the model |

## Components

The `api` package stays small and layered; each unit is independently testable.

### `settings.py` — configuration

Extend the existing `pydantic-settings` `Settings` (env prefix `TEMPORAL_API_`):

| Setting | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/models/model.zip` | path to the packaged model |
| `DEVICE` | `None` (auto cuda→mps→cpu) | torch device override |
| `CALIBRATOR_THRESHOLD` | `None` (use packaged value) | server-side override of the calibrator (logistic) decision threshold, a probability in `[0, 1]`; out-of-range fails startup; ignored (warned) for uncalibrated packages |
| `S3_BUCKET` | `""` (optional default) | default bucket holding the frames; a request's `bucket` overrides it. A request with neither fails `400` |
| `S3_REGION` | `None` | region (e.g. `gra` for OVH) |
| `S3_ENDPOINT_URL` | `None` | empty = real AWS; set for OVH / MinIO |
| `HOST` | `0.0.0.0` | bind host |
| `PORT` | `8000` | bind port |

AWS/OVH/MinIO credentials come from boto3's standard chain (env vars / IAM
role); they are never accepted in the request.

### `s3.py` — frame fetching

- A thin boto3 S3 client built from settings (`endpoint_url`, `region`).
- `fetch_frames(s3_client, bucket, keys, dest_dir) -> list[Path]`: downloads each
  object from `bucket` to a per-request temp directory **preserving the key's
  basename**, returns local paths in the **same order** as `keys`. (Key/bucket
  scheme validation happens at the request-schema layer.)
- Maps S3 errors: missing key → `frame_not_found` (404); connection/other →
  `s3_unavailable` (502).
- Temp directory is cleaned up after the response (context manager).

### `model_runner.py` — model lifecycle + inference

- Loads `model.zip` once via `BboxTubeTemporalModel.from_package(MODEL_PATH,
  device=DEVICE)`; reads `manifest.variant` / `manifest.model_version` for the
  `model` block.
- Holds the singleton and an `asyncio.Lock` (or threading lock); runs
  `predict_sequence` in a threadpool so the event loop stays free, serialized to
  respect GPU non-reentrancy.
- `predict(paths: list[Path]) -> TemporalModelOutput`.

### `schemas.py` — request/response DTOs

- Pydantic models for the request (`PredictRequest`) and the reshaped responses
  (`PredictResponse`, nested `Details`, `Tube`, `TubeEntry`, `Decision`,
  `Preprocessing`, `Model`).
- A mapper `to_response(out: TemporalModelOutput, model_meta, *, verbose: bool)
  -> PredictResponse` that reshapes `TemporalModelOutput` + `BboxTubeDetails`
  into the contract above and computes top-level `probability` per the rule.

### `app.py` — FastAPI wiring

- `lifespan` loads the model into app state; `/health` reports readiness.
- `POST /predict` (with `verbose: bool = False` query param): fetch frames →
  run model → map to response.
- Exception handlers translate domain errors into the error table above.

## Data flow

```
client ──POST /predict {frames:[keys]}──► app.py
  app.py ──keys──► s3.fetch_frames ──► temp dir (basenames preserved, ordered) ──► [Path]
  app.py ──[Path]──► model_runner.predict (lock + threadpool)
            └─► BboxTubeTemporalModel.predict_sequence
                  └─► YOLO → tubes → patches → temporal classifier → calibrator
  TemporalModelOutput ──► schemas.to_response(verbose) ──► JSON ──► client
  (temp dir cleaned up)
```

## Model package format (shared contract)

The model-package format (the `manifest.yaml` schema) is a **contract shared
between `train` (writer) and `api`/`core` (reader)** and therefore belongs in
`core`, not `api`. This spec depends on one **additive** change to that schema:

- Add an optional **`model_version`** field to the manifest, stamped by the
  training/packaging step. The API reads `manifest.get("model_version")` and
  surfaces it as `model.version`; **absent → `null`** (existing `vision-rd`
  packages predate the field and remain loadable).

The versioning scheme itself (semver in `params.yaml` vs training git SHA vs DVC
experiment id) is a **training-pipeline decision deferred to the training spec**.
No other manifest change is required by the API.

## Testing

Two tiers, both runnable without GPU or AWS:

1. **Fast / hermetic (default CI):**
   - **Model mocked** — patch `model_runner` to return canned
     `TemporalModelOutput` values, asserting request → S3 → reshape → response
     wiring (default and `?verbose=true`), the `probability` rule (smoke,
     non-smoke, no-tubes, uncalibrated), and every error-table row.
   - **S3 via `moto`** (`@mock_aws`) — create a fake bucket, `put_object` JPEG
     fixtures, hit `POST /predict` through FastAPI `TestClient`. No network.
2. **Slow integration (opt-in):** gated on a real `model.zip` being present
   (skip otherwise). Loads the package and runs `predict_sequence` end-to-end on
   a bundled sample sequence; asserts the verdict shape (not exact scores).

Manual/local dev: a `minio` service in `docker-compose.yml` plus
`TEMPORAL_API_S3_ENDPOINT_URL=http://minio:9000` and a one-time seed of a sample
sequence; then `curl` the API.

## Non-goals (v1)

- Raw-frame upload (multipart or base64). S3 keys only; revisit when a non-S3
  caller exists.
- Per-request credentials. (Per-request *bucket* is now supported via the
  `bucket` field; credentials still come solely from the boto3 chain.)
- Async jobs / queueing / batching across requests.
- Client-supplied (per-request) decision threshold override (decision stays
  server-side; consumers re-threshold using returned `probability`/`logit`). Note:
  a *server-side operator* override via `TEMPORAL_API_CALIBRATOR_THRESHOLD` is
  supported — it sets one threshold for the whole deployment at startup and is
  distinct from a per-request override.
- A `GET /model` descriptor endpoint or architecture fields in responses.
- Defining the `model_version` scheme (training spec) or migrating the `core`
  inference path.

## Success criteria

1. `POST /predict` with S3 keys returns the default DTO; `?verbose=true` adds the
   `details` block matching the contract.
2. Frame order and key basenames are preserved into `predict_sequence`.
3. `probability` follows the rule (number for calibrated incl. non-smoke and
   no-tubes=`0.0`; `null` only when uncalibrated).
4. The error table is enforced (each row has a test).
5. The same code path works against AWS, OVH, and MinIO by config alone.
6. Fast test tier passes with the model mocked and S3 via `moto` — no GPU, no
   network.
7. `GET /health` reflects model readiness.
