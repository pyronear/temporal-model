# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`.

## Endpoints

- `GET /health` — readiness + loaded model name/version.
- `POST /predict` — body `{ "frames": ["<s3-key>", ...] }` (ordered S3 keys);
  returns `{ is_smoke, probability, trigger_frame_index, model }`.
  `POST /predict?verbose=true` adds a `details` block (decision, preprocessing,
  per-tube tracks). See `docs/specs/2026-06-02-api-service-design.md` for the
  full contract.

## Run

```bash
make serve                  # local dev, http://localhost:8000
docker compose up --build   # API + MinIO (S3) locally
```

Configuration via env vars (prefix `TEMPORAL_API_`): `MODEL_PATH`, `DEVICE`,
`S3_BUCKET`, `S3_REGION`, `S3_ENDPOINT_URL` (empty = real AWS; set for OVH or
MinIO), `HOST`, `PORT`. AWS/OVH/MinIO credentials come from the standard boto3
chain (env vars / IAM role).

## Test

```bash
make test                   # fast, hermetic (model mocked, S3 via moto)
TEMPORAL_API_TEST_MODEL_PATH=/path/to/model.zip make test   # + integration
```
