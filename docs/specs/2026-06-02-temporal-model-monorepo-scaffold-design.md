# Temporal Model Monorepo — Scaffold Design

**Date:** 2026-06-02
**Status:** Approved (scaffold scope)
**Scope:** Directory/skeleton scaffold only. No model, training, eval, or inference logic is migrated yet — modules are stubs.

## Goal

Create a self-contained monolithic repository (`temporal-model`) that will eventually hold all code to **train**, **eval**, and **serve (API)** the bbox-tube temporal smoke classifier. This document covers only the initial scaffold: directory layout, packaging, tooling, and placeholder modules.

## Background

Today the temporal model is spread across three places in `vision-rd`:

- `lib/pyrocore` — generic `TemporalModel` protocol + types.
- `lib/bbox-tube-temporal` — the model library: tube building, patch extraction, timm-backbone temporal classifier, inference padding, packaging, logistic calibrator, and `BboxTubeTemporalModel`.
- `experiments/temporal-models/bbox-tube-temporal` — DVC pipeline, 11 training variants, eval scripts, packaging, notebooks. Depends on both libs via local editable installs.

There is **no API/serving layer** today; deployment is "load the packaged `model.zip` and call `predict_sequence()`".

This new repo consolidates those concerns into one repo with a serving layer. **No code is migrated in this scaffold step** — migration is deferred to a later effort.

## Decisions

| Decision | Choice |
|---|---|
| Code migration now | None — empty scaffold, stub modules only |
| Structure | Four independent packages at repo root: `core`, `train`, `eval`, `api` |
| Root pyproject | None — each package is self-contained |
| Import namespace | PEP 420 namespace package `temporal_model`; import as `temporal_model.core`, `.train`, `.eval`, `.api` |
| Shared-code deps | `train`/`eval`/`api` depend on `core` via `uv` path source (`{ path = "../core", editable = true }`) |
| API stack | FastAPI + uvicorn; packaged as a runnable Docker service |
| Data layers | Full kedro set in both `train` and `eval`: `01_raw, 03_primary, 05_model_input, 06_models, 07_model_output, 08_reporting` |
| DVC | `uv run dvc init --subdir` in `train` and `eval` now (subdir flag because they live under one git repo) |
| Tooling | `uv`, ruff (line-length 88, select `E,F,I,W,UP,B,SIM,PLC0415`), pytest, per-package Makefile, Python ≥ 3.11 — mirroring vision-rd |
| Spec location | `docs/specs/` |

## Directory layout

```
temporal-model/
├── LICENSE                         # exists
├── README.md                       # NEW — monorepo overview, how the 4 pkgs relate
├── .gitignore                      # NEW
├── Makefile                        # NEW — fans out install/lint/format/test to each pkg
├── docs/specs/                     # design docs
│
├── core/                           # dist: temporal-model-core
│   ├── pyproject.toml
│   ├── README.md
│   ├── Makefile
│   ├── src/temporal_model/core/    # NO __init__.py at src/temporal_model/ (namespace)
│   │   ├── __init__.py
│   │   ├── types.py                # stub: Tube, TubeEntry, …
│   │   ├── tubes.py                # stub: greedy IoU linker
│   │   ├── model_input.py          # stub: patch extraction
│   │   ├── inference.py            # stub: padding strategies
│   │   └── model.py                # stub: TemporalModel implementation
│   └── tests/test_smoke.py
│
├── train/                          # dist: temporal-model-train
│   ├── pyproject.toml              # dep: temporal-model-core (path = "../core")
│   ├── README.md
│   ├── Makefile
│   ├── params.yaml
│   ├── dvc.yaml
│   ├── .dvc/                        # from dvc init --subdir
│   ├── .dvcignore
│   ├── data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}/.gitkeep
│   ├── src/temporal_model/train/
│   │   ├── __init__.py
│   │   └── train.py                # stub CLI entry point
│   └── tests/test_smoke.py
│
├── eval/                           # dist: temporal-model-eval
│   ├── pyproject.toml              # dep: temporal-model-core
│   ├── README.md
│   ├── Makefile
│   ├── params.yaml
│   ├── dvc.yaml
│   ├── .dvc/
│   ├── .dvcignore
│   ├── data/{01_raw,03_primary,05_model_input,06_models,07_model_output,08_reporting}/.gitkeep
│   ├── src/temporal_model/eval/
│   │   ├── __init__.py
│   │   └── evaluate.py             # stub CLI entry point
│   └── tests/test_smoke.py
│
└── api/                            # dist: temporal-model-api
    ├── pyproject.toml              # dep: temporal-model-core, fastapi, uvicorn, pydantic-settings
    ├── README.md
    ├── Makefile
    ├── Dockerfile                  # uv-based build; runs uvicorn temporal_model.api.app:app
    ├── .dockerignore
    ├── docker-compose.yml          # docker compose up → serves locally
    ├── src/temporal_model/api/
    │   ├── __init__.py
    │   ├── app.py                  # FastAPI app: GET /health + POST /predict (stub)
    │   └── settings.py             # pydantic-settings config stub
    └── tests/test_app.py
```

## Component details

### Namespace packaging (PEP 420)

Each package ships part of the `temporal_model` namespace. The key rule: **there is no `src/temporal_model/__init__.py`** in any package — only the sub-package (`core/__init__.py`, `train/__init__.py`, …) has one. hatchling is configured per package with `packages = ["src/temporal_model"]`; because the namespace dir has no `__init__.py`, the distributions compose at install time. All four can be installed into the same environment without clobbering each other.

### `core` package

Stub modules with docstrings and signatures that `raise NotImplementedError` (functions) or are empty class/`...` bodies (types), so the package imports cleanly and `test_smoke.py` (an `import temporal_model.core` check) passes. Module names mirror the eventual migration target from `lib/bbox-tube-temporal`: `types`, `tubes`, `model_input`, `inference`, `model`.

### `train` / `eval` packages

- Depend on `core` via `[tool.uv.sources] temporal-model-core = { path = "../core", editable = true }`.
- `params.yaml`: skeleton with top-level keys mirroring vision-rd (`truncate`, `tubes`, `build_tubes`, `model_input`, `train_*`, `augment`, `package`) but values are placeholders / commented to signal "fill in on migration".
- `dvc.yaml`: a single placeholder stage (e.g. `noop` echoing a TODO) so the file is valid; real stages added on migration.
- `data/`: full kedro layer set, each with a `.gitkeep`.
- `dvc init --subdir` run in each so `.dvc/` + `.dvcignore` exist.
- Stub CLI (`train.py` / `evaluate.py`) with an `argparse`/`main()` skeleton that prints "not implemented".

### `api` package

- FastAPI app in `app.py`: `GET /health` returns `{"status": "ok"}`; `POST /predict` is a stub returning `501`/placeholder with a Pydantic request/response model sketch.
- `settings.py`: `pydantic-settings` `Settings` (model path, host, port) with env defaults.
- `Dockerfile`: multi-stage uv build, copies `core` + `api`, exposes the port, `CMD` runs uvicorn.
- `docker-compose.yml`: builds the image, maps the port, mounts a model volume placeholder.
- `tests/test_app.py`: FastAPI `TestClient` hitting `/health`.

### Root files

- `README.md`: explains the monorepo, the four packages, how they relate, and quick-start per package.
- `Makefile`: `install`, `lint`, `format`, `test` targets that loop `make -C <pkg> <target>` over the four packages.
- `.gitignore`: Python + uv + DVC + data-output ignores (mirroring vision-rd).

## Non-goals (this scaffold)

- Migrating any real model / training / eval / inference logic.
- Real DVC pipeline stages.
- Real `/predict` implementation or model loading.
- CI configuration.
- Pulling `pyrocore` in (the `TemporalModel` protocol decision is deferred to migration).

## Success criteria

1. `uv sync` succeeds in each of the four packages.
2. `make test` at the root passes (smoke tests: imports + `/health`).
3. `make lint` at the root passes.
4. `docker compose build` succeeds for the api package (and `docker compose up` serves `/health`).
5. `.dvc/` exists in `train` and `eval`.
6. No `src/temporal_model/__init__.py` anywhere (namespace integrity).
```
