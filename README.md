# temporal-model

Monolithic repository for the Pyronear **bbox-tube temporal smoke classifier**:
train it, evaluate it, and serve it behind an API.

> Scaffold stage: directory structure and tooling only. The model, training,
> evaluation, and inference logic are migrated from
> `vision-rd/lib/bbox-tube-temporal` and the bbox-tube-temporal experiment in a
> later step.

## Packages

Four independent packages, each with its own `pyproject.toml` and `tests/`.
They share one PEP 420 namespace package, `temporal_model`.

| Path | Distribution | Import | Purpose |
|------|--------------|--------|---------|
| `core/`  | `temporal-model-core`  | `temporal_model.core`  | Model, tube building, patch extraction, inference. |
| `train/` | `temporal-model-train` | `temporal_model.train` | DVC training pipeline. Depends on `core`. |
| `eval/`  | `temporal-model-eval`  | `temporal_model.eval`  | DVC evaluation pipeline. Depends on `core`. |
| `api/`   | `temporal-model-api`   | `temporal_model.api`   | FastAPI serving layer, shipped as a Docker service. Depends on `core`. |

`train`/`eval`/`api` depend on `core` via a `uv` path source
(`temporal-model-core = { path = "../core", editable = true }`).

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

## Layout

- `docs/specs/` — design documents.
- `docs/plans/` — implementation plans.
