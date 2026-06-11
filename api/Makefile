.PHONY: install lint format test serve gpu-setup run-gpu

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

# Benchmark/dev only — see the README's "GPU" section. torch/torchvision are
# pinned to the uv.lock releases; keep them in sync when bumping the lock.
gpu-setup: ## swap venv torch for CUDA wheels (one-time; see README "GPU")
	uv sync
	uv pip install --reinstall-package torch --reinstall-package torchvision \
		torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130

run-gpu: ## serve the API on the GPU (run gpu-setup first)
	uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not available - run: make gpu-setup'"
	uv run --no-sync uvicorn temporal_model.api.app:app --host 0.0.0.0 --port 8000
