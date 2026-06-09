# 🕐🔥 temporal-model

Monolithic repository for the Pyronear **temporal smoke classifier**:
train it, evaluate it, and serve it behind an API.

The model is a per-tube smoke classifier: a YOLO detector proposes boxes, boxes
are linked into temporal tubes (greedy IoU), each tube's frames are cropped to
224×224 patches and scored by a **ViT (DINOv2) backbone + transformer head**
that emits one logit per tube.

## Packages

Four independent packages, each with its own `pyproject.toml` and `tests/`.

| Path | Import | Purpose |
|------|--------|---------|
| `core/`  | `temporal_model.core`  | Model, tube building, patch extraction, inference, packaging. |
| `train/` | `temporal_model.train` | DVC training pipeline. Depends on `core`. |
| `eval/`  | `temporal_model.eval`  | DVC evaluation pipeline (packaged-model protocol metrics). Depends on `core`. |
| `api/`   | `temporal_model.api`   | FastAPI serving layer, shipped as a Docker service. Depends on `core`. |

`train`/`eval`/`api` depend on `core` via a `uv` path source
(`temporal-model-core = { path = "../core", editable = true }`). `core` and
`train` pull in PyTorch / timm / ultralytics, so their first `uv sync` is large.

## Quick start

```bash
make install        # uv sync across all four packages
make test           # pytest across all four packages
make lint           # ruff check across all four packages
```

Per package, `cd <pkg> && make install|lint|format|test`.

### Run the API locally (Docker)

```bash
make fetch-model   # download the released model.zip from HuggingFace (no creds)
make serve         # API + MinIO via docker compose, http://localhost:8000 (GET /health)
```

`serve` is equivalent to `cd api && docker compose up --build` and refuses to
start until `api/models/model.zip` exists, so run `make fetch-model` first (it
pulls v0.1.0 from the public HuggingFace repo; override with
`make fetch-model MODEL_VERSION=x.y.z`). The compose stack ships local-dev MinIO
defaults (bucket `frames`, `minioadmin` creds); with the model present
`/health` reports `model_loaded: true`.

## Origin

Ported from the Pyronear [`vision-rd`](https://github.com/pyronear/vision-rd)
research repo's `bbox-tube-temporal` work:

- `core/` — from [`lib/bbox-tube-temporal`](https://github.com/pyronear/vision-rd/tree/main/lib/bbox-tube-temporal)
- `train/` and `eval/` — from [`experiments/temporal-models/bbox-tube-temporal`](https://github.com/pyronear/vision-rd/tree/main/experiments/temporal-models/bbox-tube-temporal)
