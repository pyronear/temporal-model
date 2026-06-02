# temporal-model-api

FastAPI serving layer for the temporal smoke classifier, packaged as a
Docker service.

Import as `temporal_model.api`. Depends on `temporal-model-core`. Scaffold stage —
`GET /health` works; `POST /predict` is a stub (returns 501).

## Run

```bash
make serve                  # local dev, http://localhost:8000
docker compose up --build   # containerized
```

Configuration via env vars (prefix `TEMPORAL_API_`): `TEMPORAL_API_MODEL_PATH`,
`TEMPORAL_API_HOST`, `TEMPORAL_API_PORT`.
