# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version + API code version.
- `POST /predict` — body `{ "frames": [...], "source": "s3" | "local",
  "bucket": "<name>", "roi_xyxyn": [x_min, y_min, x_max, y_max],
  "detections": [[{"xyxyn": [...], "confidence": 0.6}], []] }`
  (ordered frames; `source` optional, falls back to `FRAME_SOURCE` — with
  `s3`, frames are S3 keys and `bucket` optionally overrides `S3_BUCKET`;
  with `local`, frames are relative paths under `FRAMES_ROOT` and `bucket`
  is invalid; `roi_xyxyn` optional normalized region of interest — tubes
  with no real detection intersecting it are dropped before scoring;
  `detections` optional caller-supplied boxes, one list per frame
  index-aligned with `frames`, `[]` = that frame's detector saw nothing —
  skips the bundled YOLO and its cache entirely, tubes are built from the
  supplied boxes);
  returns `{ is_smoke, probability, version }` (`probability` = max kept-tube
  calibrated probability, `null` if uncalibrated).
  `version` is `{api, model}` — the code release (== the Docker image tag,
  `null` on non-release builds) and the packaged model release.
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). `POST /predict?compute_trigger=true` runs the
  first-crossing search (extra classifier work, off by default) and adds a
  top-level `trigger_frame_index` (`null` if nothing crossed) — with
  `verbose=true` it also fills `details.decision.trigger_tube_id` and
  per-tube `details.tubes[].first_crossing_frame`. See
  `docs/specs/2026-06-02-api-service-design.md`,
  `docs/specs/2026-06-11-api-local-frames-design.md` and
  `docs/specs/2026-06-11-api-supplied-detections-design.md` for the full
  contract.

## Run

```bash
make fetch-model            # download model.zip from HuggingFace (no creds), run from repo root
make serve                  # local dev, http://localhost:8000
docker compose up --build   # API + MinIO (S3) locally
```

Run `make fetch-model` (from the repo root) before serving — it downloads the
released `model.zip` from the public HuggingFace repo into `api/models/`. The
container mounts `./models:/models` and loads `/models/model.zip`; `make serve`
refuses to start if the file is missing. (`docker compose up --build` directly
will fail at the `COPY` step without it.)

Configuration via env vars (prefix `TEMPORAL_API_`): `MODEL_PATH`, `DEVICE`,
`CALIBRATOR_THRESHOLD`, `TOKEN`, `FRAME_SOURCE`, `FRAMES_ROOT`, `S3_BUCKET`,
`S3_REGION`, `S3_ENDPOINT_URL` (empty = real AWS; set for OVH or MinIO).
AWS/OVH/MinIO credentials come from the standard boto3 chain
(env vars / IAM role). `S3_BUCKET` is an optional default; a request may
override it per call with its `bucket` field (needed for alert-api stacks
whose per-org bucket names are not known ahead of time). A request with
neither is rejected with `400 invalid_request`.

`FRAME_SOURCE` (default `s3`) selects where `/predict` frames come from when
a request omits its optional `source` field. With `local` (an edge box whose
frames sit on a shared volume), `frames` are relative paths resolved under
`FRAMES_ROOT`; `FRAMES_ROOT` is settings-only by design — a request-supplied
root would let callers probe arbitrary server paths — and absolute paths or
`..` segments are rejected with `400 invalid_request`. A missing file is the
same `404 frame_not_found` as a missing S3 key, and local requests skip the
S3 download entirely (frames are read in place). `FRAME_SOURCE=local` without
`FRAMES_ROOT` fails at startup; a root that is not a directory at request
time (typo, unmounted volume) is a distinct `400`, not a per-frame 404.

Two invariants for local frame producers: frames are read in place at predict
time, so publish them atomically (write to a temp name, then rename) — a
frame mid-write can fail the request or score a truncated image. And when
the detection cache is enabled, frame basenames (stems) must stay globally
unique across cameras and time — the same invariant S3 keys already carry
(the cache is keyed by stem; see `detection_cache.py`).

`CALIBRATOR_THRESHOLD` (a probability in `[0, 1]`) overrides the packaged
calibrator decision threshold for every prediction; out-of-range values fail
startup, and it is ignored (with a warning) for uncalibrated packages. With
`?verbose=true`, the response's `details.decision` reports `threshold_overridden`
and the original `packaged_threshold`.

`TOKEN` (optional, use an ASCII value) guards `POST /predict`: when set, callers
must send `Authorization: Bearer <token>` or receive `401 unauthorized`. When
unset, auth is disabled (the API logs a warning at startup) and `/predict` is
open. `GET /health` is never guarded, so load balancers can probe it without a
token.

### GPU (benchmark/dev only)

The published Docker image is CPU-only by design: torch is pinned to the
`pytorch-cpu` wheel index (`pyproject.toml`) to keep the image small, so the
container cannot use CUDA even on a GPU host. To serve on a GPU, run natively
from `api/`:

```bash
make gpu-setup              # one-time: swap venv torch for CUDA wheels (multi-GB)
TEMPORAL_API_MODEL_PATH=$PWD/models/model.zip make run-gpu
```

`gpu-setup` replaces the venv's torch/torchvision with CUDA wheels (`cu130`
index, pinned to the `uv.lock` releases — needs an NVIDIA driver supporting
CUDA 13, i.e. >= 580). The venv then diverges from `uv.lock`: any plain
`uv run` or `uv sync` — including `make test`/`make lint`/`make format` —
restores the locked CPU wheels. `run-gpu` serves with `uv run --no-sync` and
refuses to start when the CUDA wheels are gone, so rerun `make gpu-setup`
after one of those (cheap once the wheels are cached).

Unlike the Docker flow, `MODEL_PATH` must be set: the `/models/model.zip`
default only exists inside the container (`make fetch-model` from the repo
root downloads to `api/models/`). The model auto-detects `cuda` when
`TEMPORAL_API_DEVICE` is unset; the S3 env vars apply as above. `run-gpu`
binds `0.0.0.0:8000`, so set `TEMPORAL_API_TOKEN` on shared networks —
without it `/predict` is open to anyone who can reach the host. The compose
`api` service publishes the same port: when using compose just for S3, start
only the pieces you need (`docker compose up -d minio createbuckets`).

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
