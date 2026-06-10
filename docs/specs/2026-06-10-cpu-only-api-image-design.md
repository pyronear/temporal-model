# CPU-Only API Serving Image — Design

**Date:** 2026-06-10
**Status:** Implemented
**Scope:** Shrink the `api` Docker image by dropping the unused NVIDIA CUDA
stack. The API deploys on CPU only (GPU is dev/training territory), so the
CUDA build of torch is dead weight. Training (`train`) keeps its GPU torch.

## Problem

The default PyPI `torch` wheel on Linux declares the full NVIDIA CUDA runtime
as dependencies. In the serving image that meant **2.7 GB of `nvidia-*` libs**
plus a **1.2 GB CUDA torch build** the CPU box never touches. The `uv sync`
layer alone was 5.54 GB; the whole image was 6.27 GB.

## Change

1. **Pin torch/torchvision to the PyTorch CPU index** in `api/pyproject.toml`:

   ```toml
   [tool.uv.sources]
   torch = { index = "pytorch-cpu" }
   torchvision = { index = "pytorch-cpu" }

   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true
   ```

   `torch`/`torchvision` are declared by `core`, and a uv source override on a
   *transitively*-pulled package is silently ignored — so they are also added
   as **direct `api` dependencies** to bind the override. Each project
   (`api`, `core`, `train`, `eval`, `benchmark`) has its own lockfile and
   resolves independently, so **GPU training is unaffected**.

2. **Stop double-baking `model.zip`** in `api/Dockerfile`. `COPY api/ ./api/`
   pulled in the dockerignore-whitelisted `api/models/model.zip` (163 MB) on
   top of the explicit `COPY api/models/model.zip /models/model.zip`. Replaced
   with a targeted copy of only `pyproject.toml`, `uv.lock`, and `src/`
   (runtime reads the model from `/models/model.zip`, per the settings default;
   the in-tree copy was never used).

## Result

| | Before | After |
|---|---|---|
| Image size | 6.27 GB | **2.05 GB** (−67%) |
| `uv sync` venv layer | 5.54 GB | 1.49 GB |
| NVIDIA CUDA libs | 2.7 GB | 0 |
| torch | 1.2 GB (CUDA) | 698 MB (CPU) |

## Verification

- 76 api tests pass (1 skipped) on `torch 2.12.0+cpu` (`cuda_available=False`),
  including the model-loading and S3 integration tests that exercise `core`'s
  inference path.
- Image builds clean; lock has 0 nvidia packages, triton removed.
- Container boots and `/health` reports `model_loaded: true`.

## Non-goals

- A GPU serving image variant (YAGNI — the API only deploys on CPU). The CPU
  pin composes cleanly into a multi-target Dockerfile later if that changes.
- Multi-stage builds / uv cache trimming (marginal vs. the torch win).
