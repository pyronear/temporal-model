.PHONY: install lint format test import

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v

import: ## import new sequences from alert-api, then dvc add + push the store
	uv run temporal-monitor import $(ARGS)
	uv run dvc add data/01_raw/sequences
	uv run dvc push
