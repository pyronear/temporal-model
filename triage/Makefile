.PHONY: install lint format test pull

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v

pull: ## pull unannotated sequences from the annotator (read-only), then dvc add + push the store
	uv run temporal-triage pull $(ARGS)
	uv run dvc add data/01_raw/sequences
	uv run dvc push
