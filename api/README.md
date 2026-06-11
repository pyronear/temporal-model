# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version + API code version.
- `POST /predict` — body `{ "frames": ["<s3-key>", ...], "bucket": "<name>",
  "roi_xyxyn": [x_min, y_min, x_max, y_max] }`
  (ordered S3 keys; `bucket` optional, falls back to `S3_BUCKET`;
  `roi_xyxyn` optional normalized region of interest — tubes with no real
  detection intersecting it are dropped before scoring);
  returns `{ is_smoke, probability, version }` (`probability` = max kept-tube
  calibrated probability, `null` if uncalibrated).
  `version` is `{api, model}` — the code release (== the Docker image tag,
  `null` on non-release builds) and the packaged model release.
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). See `docs/specs/2026-06-02-api-service-design.md` for the
  full contract.

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
`CALIBRATOR_THRESHOLD`, `TOKEN`, `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL`
(empty = real AWS; set for OVH or MinIO), `HOST`, `PORT`. AWS/OVH/MinIO credentials come from
the standard boto3 chain (env vars / IAM role). `S3_BUCKET` is an optional
default; a request may override it per call with its `bucket` field (needed for
alert-api stacks whose per-org bucket names are not known ahead of time). A
request with neither is rejected with `400 invalid_request`.

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
container cannot use CUDA even on a GPU host. To serve on a GPU, run natively:

```bash
make run-gpu                # binds 0.0.0.0:8000, model auto-picks cuda
```

This swaps the venv's torch/torchvision for CUDA wheels (`cu130` index — needs
an NVIDIA driver supporting CUDA 13, i.e. >= 580) and serves with
`uv run --no-sync`. The venv then diverges from `uv.lock`: any plain `uv run`
or `uv sync` restores the locked CPU wheels (rerun `make run-gpu` to flip
back — wheels are cached after the first multi-GB download). The model
auto-detects `cuda` when `TEMPORAL_API_DEVICE` is unset; the usual env vars
(`MODEL_PATH`, `S3_*`, …) apply as above.

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
