.PHONY: install lint format test serve run-gpu

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

# Benchmark/dev only — the released Docker image stays CPU-only on purpose.
# Swaps the venv's torch/torchvision for CUDA wheels, then serves with
# --no-sync (a plain `uv run`/`uv sync` reverts the venv to the locked CPU
# wheels). Needs an NVIDIA driver supporting CUDA 13 (>= 580).
run-gpu: ## run the API on a CUDA GPU (swaps venv torch for CUDA wheels)
	uv sync
	uv pip install --reinstall-package torch --reinstall-package torchvision \
		torch torchvision --index-url https://download.pytorch.org/whl/cu130
	uv run --no-sync uvicorn temporal_model.api.app:app --host 0.0.0.0 --port 8000
