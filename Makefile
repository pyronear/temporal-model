PACKAGES := core train eval api benchmark

# Released model.zip version fetched from HuggingFace by `fetch-model`.
# Pinned in api/MODEL_VERSION — the repo version and the model version are
# decoupled (code can change without retraining).
MODEL_VERSION ?= $(shell cat api/MODEL_VERSION)
MODEL_ZIP := api/models/model.zip

.DEFAULT_GOAL := help
.PHONY: help install lint format test serve fetch-model

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	    | sort \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## uv sync every package
	@fail=0; for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg install || fail=1; done; exit $$fail

lint: ## ruff check every package + docs scripts
	@fail=0; for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg lint || fail=1; done; exit $$fail
	@echo "==> docs/assets/scripts"
	uv run --project core ruff check docs/assets/scripts

format: ## ruff format every package + docs scripts
	@fail=0; for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg format || fail=1; done; exit $$fail
	@echo "==> docs/assets/scripts"
	uv run --project core ruff format docs/assets/scripts

test: ## pytest every package
	@fail=0; for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg test || fail=1; done; exit $$fail

fetch-model: ## download the released model.zip from HuggingFace (no creds)
	cd api && uv run python -m temporal_model.api.release \
	    fetch --version $(MODEL_VERSION) --output models/model.zip

serve: ## run the full API + MinIO stack locally (docker compose)
	@test -f $(MODEL_ZIP) || { \
	    echo "$(MODEL_ZIP) not found — run 'make fetch-model' (downloads v$(MODEL_VERSION) from HuggingFace, no credentials)"; \
	    exit 1; \
	}
	cd api && docker compose up --build
