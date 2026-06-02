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
