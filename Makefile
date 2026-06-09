PACKAGES := core train eval api

# Released model.zip version fetched from HuggingFace by `fetch-model`.
MODEL_VERSION ?= 0.1.0
MODEL_ZIP := api/models/model.zip

.PHONY: help install lint format test serve fetch-model

help: ## Show this help
	@echo "Fans out targets across: $(PACKAGES)"
	@echo "Targets: install lint format test"
	@echo "API only: fetch-model (download model.zip from HF), serve (API + MinIO via docker compose)"

install: ## uv sync every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg install; done

lint: ## ruff check every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg lint; done

format: ## ruff format every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg format; done

test: ## pytest every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg test; done

fetch-model: ## download the released model.zip from HuggingFace (no creds)
	cd api && uv run python -m temporal_model.api.release \
	    fetch --version $(MODEL_VERSION) --output models/model.zip

serve: ## run the full API + MinIO stack locally (docker compose)
	@test -f $(MODEL_ZIP) || { \
	    echo "$(MODEL_ZIP) not found — run 'make fetch-model' (downloads v$(MODEL_VERSION) from HuggingFace, no credentials)"; \
	    exit 1; \
	}
	cd api && docker compose up --build
