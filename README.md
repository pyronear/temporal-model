# temporal-model

Monolithic repository for the Pyronear **temporal smoke classifier**:
train it, evaluate it, and serve it behind an API.

The model is a per-tube smoke classifier: a YOLO detector proposes boxes, boxes
are linked into temporal tubes (greedy IoU), each tube's frames are cropped to
224×224 patches and scored by a **ViT (DINOv2) backbone + transformer head**
that emits one logit per tube. The repo is scoped to the production
`vit_dinov2_finetune` model.

## Packages

Four independent packages, each with its own `pyproject.toml` and `tests/`.
They share one PEP 420 namespace package, `temporal_model`.

| Path | Distribution | Import | Purpose |
|------|--------------|--------|---------|
| `core/`  | `temporal-model-core`  | `temporal_model.core`  | Model, tube building, patch extraction, inference, packaging. |
| `train/` | `temporal-model-train` | `temporal_model.train` | DVC training pipeline. Depends on `core`. |
| `eval/`  | `temporal-model-eval`  | `temporal_model.eval`  | DVC evaluation pipeline (packaged-model protocol metrics). Depends on `core`. |
| `api/`   | `temporal-model-api`   | `temporal_model.api`   | FastAPI serving layer, shipped as a Docker service. Depends on `core`. |

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
cd api
docker compose up --build      # serves http://localhost:8000 (GET /health)
```
