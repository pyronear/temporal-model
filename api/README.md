# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version.
- `POST /predict` — body `{ "frames": ["<s3-key>", ...] }` (ordered S3 keys);
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
`CALIBRATOR_THRESHOLD`, `S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL` (empty = real
AWS; set for OVH or MinIO), `HOST`, `PORT`. AWS/OVH/MinIO credentials come from
the standard boto3 chain (env vars / IAM role).

`CALIBRATOR_THRESHOLD` (a probability in `[0, 1]`) overrides the packaged
calibrator decision threshold for every prediction; out-of-range values fail
startup, and it is ignored (with a warning) for uncalibrated packages. With
`?verbose=true`, the response's `details.decision` reports `threshold_overridden`
and the original `packaged_threshold`.

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
