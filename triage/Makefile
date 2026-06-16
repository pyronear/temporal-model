.PHONY: install lint format test pull share unpack viewer

install: ## uv sync
	uv sync

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v

pull: ## pull unannotated sequences from the annotator (read-only)
	uv run temporal-triage pull $(ARGS)

share: ## pack the scored store+report into shards, then dvc add + push
	uv run temporal-triage pack
	uv run dvc add data/02_shards
	uv run dvc push

unpack: ## restore the loose store + report from pulled shards
	uv run temporal-triage unpack

viewer: ## launch the eval viewer over this triage tree
	cd ../viewer && DATA_ROOT=../triage npm run dev
