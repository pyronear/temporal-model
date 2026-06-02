# Temporal Model Monorepo Scaffold — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold a self-contained `temporal-model` monorepo with four independent packages (`core`, `train`, `eval`, `api`) — directory layout, packaging, tooling, DVC, Docker, and importable stub modules. No model/training/eval/inference logic is migrated.

**Architecture:** Four packages at the repo root, each with its own `pyproject.toml` + `tests/` (no root pyproject). All ship parts of one PEP 420 namespace package `temporal_model` (no `src/temporal_model/__init__.py` anywhere). `train`/`eval`/`api` depend on `core` via `uv` path sources. `train`/`eval` carry DVC (`dvc init --subdir`) and kedro-style `data/` layers. `api` is a FastAPI service packaged as a Docker container.

**Tech Stack:** Python ≥3.11, `uv`, hatchling, ruff, pytest, DVC, FastAPI + uvicorn, Docker.

**Spec:** `docs/specs/2026-06-02-temporal-model-monorepo-scaffold-design.md`

---

## File structure

```
temporal-model/
├── README.md  .gitignore  Makefile            # Task 1
├── core/   (pyproject, Makefile, README, src stubs, tests)        # Task 2
├── train/  (pyproject, Makefile, README, params, dvc, data, src)  # Task 3
├── eval/   (pyproject, Makefile, README, params, dvc, data, src)  # Task 4
└── api/    (pyproject, Makefile, README, app, settings, Docker)   # Task 5
                                                                   # Task 6: verify all
```

---

## Task 0: Branch

- [ ] **Step 1: Create a working branch**

Run:
```bash
git checkout -b scaffold-monorepo
```

- [ ] **Step 2: Commit the already-written design docs**

```bash
git add docs/specs docs/plans
git commit -m "docs: add monorepo scaffold spec and plan"
```

---

## Task 1: Root files

**Files:**
- Create: `README.md`
- Create: `.gitignore`
- Create: `Makefile`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/

# Tooling caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.ipynb_checkpoints/

# DVC data layers — keep the directory structure, ignore the contents
**/data/**
!**/data/**/
!**/data/**/.gitkeep
**/.dvc/cache
**/.dvc/tmp

# API model artifacts mounted at runtime
api/models/

# OS
.DS_Store
```

- [ ] **Step 2: Write the root `Makefile`**

```makefile
PACKAGES := core train eval api

.PHONY: help install lint format test

help: ## Show this help
	@echo "Fans out targets across: $(PACKAGES)"
	@echo "Targets: install lint format test"

install: ## uv sync every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg install; done

lint: ## ruff check every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg lint; done

format: ## ruff format every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg format; done

test: ## pytest every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg test; done
```

- [ ] **Step 3: Write `README.md`**

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add README.md .gitignore Makefile
git commit -m "chore: add root README, gitignore, and fan-out Makefile"
```

---

## Task 2: `core` package

**Files:**
- Create: `core/pyproject.toml`
- Create: `core/Makefile`
- Create: `core/README.md`
- Create: `core/src/temporal_model/core/__init__.py`
- Create: `core/src/temporal_model/core/types.py`
- Create: `core/src/temporal_model/core/tubes.py`
- Create: `core/src/temporal_model/core/model_input.py`
- Create: `core/src/temporal_model/core/inference.py`
- Create: `core/src/temporal_model/core/model.py`
- Test: `core/tests/test_smoke.py`

> **Namespace integrity:** do NOT create `core/src/temporal_model/__init__.py`. Only the `core/` sub-package gets an `__init__.py`.

- [ ] **Step 1: Write `core/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-core"
version = "0.1.0"
description = "Core model, tube building, and inference for the bbox-tube temporal smoke classifier"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26,<2",
    "pydantic>=2.6",
]

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Write `core/Makefile`**

```makefile
.PHONY: install lint format test

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Write `core/README.md`**

```markdown
# temporal-model-core

Core building blocks of the bbox-tube temporal smoke classifier: data types,
tube linking, patch extraction, the temporal classifier, and inference.

Import as `temporal_model.core`. Scaffold stage — modules are stubs; logic is
migrated from `vision-rd/lib/bbox-tube-temporal` later.
```

- [ ] **Step 4: Write the stub modules**

`core/src/temporal_model/core/__init__.py`:
```python
"""Core model, tube-building, and inference for the bbox-tube temporal smoke classifier.

Scaffold stub. Implementations are migrated from
``vision-rd/lib/bbox-tube-temporal`` in a later step.
"""
```

`core/src/temporal_model/core/types.py`:
```python
"""Core data types: tubes, tube entries, and related structures.

Scaffold stub. Target: vision-rd/lib/bbox-tube-temporal/.../types.py.
"""
```

`core/src/temporal_model/core/tubes.py`:
```python
"""Tube building: greedy IoU linking of YOLO detections across frames.

Scaffold stub. Target: vision-rd/lib/bbox-tube-temporal/.../tubes.py.
"""
```

`core/src/temporal_model/core/model_input.py`:
```python
"""Model input: context-expanded square patch extraction and resizing.

Scaffold stub. Target: vision-rd/lib/bbox-tube-temporal/.../model_input.py.
"""
```

`core/src/temporal_model/core/inference.py`:
```python
"""Inference helpers: symmetric / uniform padding of short sequences.

Scaffold stub. Target: vision-rd/lib/bbox-tube-temporal/.../inference.py.
"""
```

`core/src/temporal_model/core/model.py`:
```python
"""Temporal model: the packaged classifier implementing the TemporalModel protocol.

Scaffold stub. Target: vision-rd/lib/bbox-tube-temporal/.../model.py.
"""
```

- [ ] **Step 5: Write the smoke test**

`core/tests/test_smoke.py`:
```python
def test_core_subpackage_imports():
    import temporal_model.core
    from temporal_model.core import (
        inference,
        model,
        model_input,
        tubes,
        types,
    )

    assert temporal_model.core is not None
    assert all(
        mod is not None
        for mod in (types, tubes, model_input, inference, model)
    )
```

- [ ] **Step 6: Sync and run the test (expect PASS)**

Run:
```bash
cd core && uv sync && uv run pytest tests/ -v
```
Expected: `test_core_subpackage_imports PASSED`.

- [ ] **Step 7: Lint**

Run:
```bash
uv run ruff check .
```
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
cd .. && git add core
git commit -m "feat(core): scaffold temporal-model-core package with stub modules"
```

---

## Task 3: `train` package

**Files:**
- Create: `train/pyproject.toml`
- Create: `train/Makefile`
- Create: `train/README.md`
- Create: `train/params.yaml`
- Create: `train/dvc.yaml`
- Create: `train/data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}/.gitkeep`
- Create: `train/src/temporal_model/train/__init__.py`
- Create: `train/src/temporal_model/train/train.py`
- Test: `train/tests/test_smoke.py`

- [ ] **Step 1: Write `train/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-train"
version = "0.1.0"
description = "DVC training pipeline for the bbox-tube temporal smoke classifier"
requires-python = ">=3.11"
dependencies = [
    "temporal-model-core",
    "pyyaml>=6.0",
]

[project.scripts]
temporal-train = "temporal_model.train.train:main"

[tool.uv.sources]
temporal-model-core = { path = "../core", editable = true }

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "dvc>=3.56",
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Write `train/Makefile`** (identical target set to core)

```makefile
.PHONY: install lint format test

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Write `train/README.md`**

```markdown
# temporal-model-train

DVC training pipeline for the bbox-tube temporal smoke classifier.

Import as `temporal_model.train`; CLI entry point `temporal-train`. Depends on
`temporal-model-core`. Scaffold stage — `dvc.yaml` holds a placeholder stage and
`train.py` is a stub.

```bash
make install
uv run dvc repro       # once real stages exist
```
```

- [ ] **Step 4: Write `train/params.yaml`** (skeleton mirroring vision-rd keys; values are placeholders to fill on migration)

```yaml
# Training pipeline parameters. Scaffold skeleton — values are placeholders.
truncate:
  max_frames: 20

tubes:
  iou_threshold: 0.2
  max_misses: 2

build_tubes:
  min_tube_length: 4

model_input:
  context_factor: 1.5
  patch_size: 224

train:
  seed: 42
  backbone: resnet18
  head: gru
  learning_rate: 1.0e-3
  batch_size: 16
  max_epochs: 30

augment:
  flip_prob: 0.5
```

- [ ] **Step 5: Write `train/dvc.yaml`** (valid placeholder stage)

```yaml
stages:
  noop:
    desc: "Placeholder stage — replace with the real training pipeline on migration."
    cmd: echo "TODO: implement truncate -> build_tubes -> build_model_input -> train stages"
```

- [ ] **Step 6: Create the data-layer directories with `.gitkeep`**

Run:
```bash
mkdir -p train/data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}
for d in train/data/*/; do touch "$d/.gitkeep"; done
```

- [ ] **Step 7: Write the stub source**

`train/src/temporal_model/train/__init__.py`:
```python
"""Training pipeline for the bbox-tube temporal smoke classifier (scaffold stub)."""
```

`train/src/temporal_model/train/train.py`:
```python
"""Training entry point.

Scaffold stub — the real DVC-driven training loop is migrated later.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the bbox-tube temporal smoke classifier."
    )
    parser.add_argument(
        "--params", default="params.yaml", help="Path to params.yaml"
    )
    parser.parse_args()
    raise SystemExit("temporal-train: not implemented yet (scaffold stub)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write the smoke test**

`train/tests/test_smoke.py`:
```python
def test_train_imports():
    from temporal_model.train import train

    assert callable(train.main)


def test_core_dependency_importable():
    import temporal_model.core

    assert temporal_model.core is not None
```

- [ ] **Step 9: Sync, init DVC (subdir), run tests**

Run:
```bash
cd train && uv sync && uv run dvc init --subdir && uv run pytest tests/ -v && uv run ruff check .
```
Expected: `uv sync` resolves `temporal-model-core` from `../core`; `dvc init --subdir` creates `train/.dvc/` + `train/.dvcignore`; both tests PASS; ruff clean.

> If `dvc init --subdir` reports the directory is already tracked, it is safe — `.dvc/` will still be created. If editable namespace resolution fails on `import temporal_model.core`, run `uv sync --reinstall-package temporal-model-core` and re-test (note in PR if needed).

- [ ] **Step 10: Commit**

```bash
cd .. && git add train
git commit -m "feat(train): scaffold temporal-model-train package with DVC + data layers"
```

---

## Task 4: `eval` package

**Files:**
- Create: `eval/pyproject.toml`
- Create: `eval/Makefile`
- Create: `eval/README.md`
- Create: `eval/params.yaml`
- Create: `eval/dvc.yaml`
- Create: `eval/data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}/.gitkeep`
- Create: `eval/src/temporal_model/eval/__init__.py`
- Create: `eval/src/temporal_model/eval/evaluate.py`
- Test: `eval/tests/test_smoke.py`

- [ ] **Step 1: Write `eval/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-eval"
version = "0.1.0"
description = "DVC evaluation pipeline for the bbox-tube temporal smoke classifier"
requires-python = ">=3.11"
dependencies = [
    "temporal-model-core",
    "pyyaml>=6.0",
]

[project.scripts]
temporal-eval = "temporal_model.eval.evaluate:main"

[tool.uv.sources]
temporal-model-core = { path = "../core", editable = true }

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "dvc>=3.56",
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Write `eval/Makefile`** (identical target set to core)

```makefile
.PHONY: install lint format test

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Write `eval/README.md`**

```markdown
# temporal-model-eval

DVC evaluation pipeline for the bbox-tube temporal smoke classifier (per-tube and
protocol-level metrics).

Import as `temporal_model.eval`; CLI entry point `temporal-eval`. Depends on
`temporal-model-core`. Scaffold stage — `dvc.yaml` holds a placeholder stage and
`evaluate.py` is a stub.
```

- [ ] **Step 4: Write `eval/params.yaml`**

```yaml
# Evaluation pipeline parameters. Scaffold skeleton — values are placeholders.
evaluate:
  threshold: 0.5
  target_recall: 0.95

protocol:
  aggregation: max_logit
```

- [ ] **Step 5: Write `eval/dvc.yaml`**

```yaml
stages:
  noop:
    desc: "Placeholder stage — replace with the real evaluation pipeline on migration."
    cmd: echo "TODO: implement evaluate -> evaluate_packaged -> compare_variants stages"
```

- [ ] **Step 6: Create the data-layer directories with `.gitkeep`**

Run:
```bash
mkdir -p eval/data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}
for d in eval/data/*/; do touch "$d/.gitkeep"; done
```

- [ ] **Step 7: Write the stub source**

`eval/src/temporal_model/eval/__init__.py`:
```python
"""Evaluation pipeline for the bbox-tube temporal smoke classifier (scaffold stub)."""
```

`eval/src/temporal_model/eval/evaluate.py`:
```python
"""Evaluation entry point.

Scaffold stub — the real DVC-driven evaluation is migrated later.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the bbox-tube temporal smoke classifier."
    )
    parser.add_argument(
        "--params", default="params.yaml", help="Path to params.yaml"
    )
    parser.parse_args()
    raise SystemExit("temporal-eval: not implemented yet (scaffold stub)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write the smoke test**

`eval/tests/test_smoke.py`:
```python
def test_eval_imports():
    from temporal_model.eval import evaluate

    assert callable(evaluate.main)


def test_core_dependency_importable():
    import temporal_model.core

    assert temporal_model.core is not None
```

- [ ] **Step 9: Sync, init DVC (subdir), run tests**

Run:
```bash
cd eval && uv sync && uv run dvc init --subdir && uv run pytest tests/ -v && uv run ruff check .
```
Expected: `eval/.dvc/` created; both tests PASS; ruff clean.

- [ ] **Step 10: Commit**

```bash
cd .. && git add eval
git commit -m "feat(eval): scaffold temporal-model-eval package with DVC + data layers"
```

---

## Task 5: `api` package (FastAPI + Docker)

**Files:**
- Create: `api/pyproject.toml`
- Create: `api/Makefile`
- Create: `api/README.md`
- Create: `api/src/temporal_model/api/__init__.py`
- Create: `api/src/temporal_model/api/app.py`
- Create: `api/src/temporal_model/api/settings.py`
- Create: `api/Dockerfile`
- Create: `api/.dockerignore`
- Create: `api/docker-compose.yml`
- Test: `api/tests/test_app.py`

- [ ] **Step 1: Write `api/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "temporal-model-api"
version = "0.1.0"
description = "FastAPI serving layer for the bbox-tube temporal smoke classifier"
requires-python = ">=3.11"
dependencies = [
    "temporal-model-core",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic-settings>=2.2",
]

[tool.uv.sources]
temporal-model-core = { path = "../core", editable = true }

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model"]

[dependency-groups]
dev = [
    "httpx>=0.27",
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.lint.isort]
known-first-party = ["temporal_model"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Write `api/Makefile`** (adds a `serve` target)

```makefile
.PHONY: install lint format test serve

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v

serve: ## run the API locally with uvicorn
	uv run uvicorn temporal_model.api.app:app --reload
```

- [ ] **Step 3: Write `api/README.md`**

```markdown
# temporal-model-api

FastAPI serving layer for the bbox-tube temporal smoke classifier, packaged as a
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
```

- [ ] **Step 4: Write `api/src/temporal_model/api/__init__.py`**

```python
"""FastAPI serving layer for the bbox-tube temporal smoke classifier (scaffold stub)."""
```

- [ ] **Step 5: Write `api/src/temporal_model/api/settings.py`**

```python
"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_API_",
        protected_namespaces=(),
    )

    model_path: str = "/models/model.zip"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
```

- [ ] **Step 6: Write `api/src/temporal_model/api/app.py`**

```python
"""FastAPI application.

Scaffold stub: ``/health`` is live; ``/predict`` returns 501 until the model
loading + inference path is migrated from ``temporal_model.core``.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Temporal Model API", version="0.1.0")


class PredictRequest(BaseModel):
    frame_paths: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": "predict not implemented yet (scaffold stub)"},
    )
```

- [ ] **Step 7: Write the test**

`api/tests/test_app.py`:
```python
from fastapi.testclient import TestClient

from temporal_model.api.app import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_stub_returns_501():
    response = client.post("/predict", json={"frame_paths": []})
    assert response.status_code == 501
```

- [ ] **Step 8: Write `api/.dockerignore`**

```dockerignore
**/.venv
**/__pycache__
**/*.pyc
**/.pytest_cache
**/.ruff_cache
**/.git
**/data
**/.dvc/cache
api/models
```

- [ ] **Step 9: Write `api/Dockerfile`** (build context is the repo root, so `core/` is available next to `api/`)

```dockerfile
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Bring in the core path-dependency and the api package itself.
COPY core/ ./core/
COPY api/ ./api/

WORKDIR /app/api
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "temporal_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 10: Write `api/docker-compose.yml`** (context is parent dir so the Dockerfile can copy `core/`)

```yaml
services:
  api:
    build:
      context: ..
      dockerfile: api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - TEMPORAL_API_MODEL_PATH=/models/model.zip
    volumes:
      - ./models:/models
```

- [ ] **Step 11: Sync and run tests**

Run:
```bash
cd api && uv sync && uv run pytest tests/ -v && uv run ruff check .
```
Expected: both tests PASS; ruff clean.

- [ ] **Step 12: Build and smoke-test the Docker image**

Run:
```bash
docker compose build
docker compose up -d
sleep 3
curl -fsS http://localhost:8000/health
docker compose down
```
Expected: build succeeds; `curl` prints `{"status":"ok"}`.

> If Docker is unavailable in this environment, skip Step 12 and note it — the pytest `/health` test already covers the app; the Docker build is verified separately.

- [ ] **Step 13: Commit**

```bash
cd .. && git add api
git commit -m "feat(api): scaffold FastAPI service with Docker packaging"
```

---

## Task 6: Whole-repo verification

- [ ] **Step 1: Run the fan-out targets from the root**

Run:
```bash
make install
make lint
make test
```
Expected: all four packages sync, lint clean, and all smoke tests PASS.

- [ ] **Step 2: Verify namespace integrity (no namespace-level `__init__.py`)**

Run:
```bash
find . -path '*/src/temporal_model/__init__.py' -not -path '*/.venv/*'
```
Expected: **no output** (the namespace dir must not have an `__init__.py`).

- [ ] **Step 3: Verify DVC initialized in train and eval**

Run:
```bash
ls -d train/.dvc eval/.dvc
```
Expected: both directories exist.

- [ ] **Step 4: Final commit (lockfiles / any remaining)**

```bash
git add -A
git commit -m "chore: finalize monorepo scaffold (lockfiles, dvc init artifacts)" || echo "nothing to commit"
```

---

## Self-review notes

- **Spec coverage:** every spec section maps to a task — root files (T1), core (T2), train+DVC+data (T3), eval+DVC+data (T4), api+Docker (T5), success criteria 1–6 (T6 + per-package sync/test steps). ✅
- **Namespace integrity** is explicitly verified (T6 Step 2) and called out as a "do not create" in T2. ✅
- **Editable namespace risk** (uv editable install of a namespace path dep) is flagged with a remediation in T3 Step 9. ✅
- **Docker context** subtlety (build from repo root so `core/` is copyable) is encoded in both the Dockerfile copy paths and compose `context: ..`. ✅
- **No placeholders** in steps — every file's full contents are given. The `params.yaml`/`dvc.yaml` "placeholder stage" content is intentional scaffold output, not a plan gap. ✅
```
