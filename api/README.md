# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version.
- `POST /predict` — body `{ "frames": ["<s3-key>", ...], "bucket": "<name>",
  "roi_xyxyn": [x_min, y_min, x_max, y_max],
  "detections": [[{"xyxyn": [...], "confidence": 0.6}], []] }`
  (ordered S3 keys; `bucket` optional, falls back to `S3_BUCKET`;
  `roi_xyxyn` optional normalized region of interest — tubes with no real
  detection intersecting it are dropped before scoring;
  `detections` optional caller-supplied boxes, one list per frame
  index-aligned with `frames`, `[]` = that frame's detector saw nothing —
  skips the bundled YOLO and its cache entirely, tubes are built from the
  supplied boxes);
  returns `{ is_smoke, probability, model }` (`probability` = max kept-tube
  calibrated probability, `null` if uncalibrated).
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

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
