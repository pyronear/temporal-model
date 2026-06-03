PACKAGES := core train eval api

.PHONY: help install lint format test serve

help: ## Show this help
	@echo "Fans out targets across: $(PACKAGES)"
	@echo "Targets: install lint format test"
	@echo "API only: serve (API + MinIO via docker compose)"

install: ## uv sync every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg install; done

lint: ## ruff check every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg lint; done

format: ## ruff format every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg format; done

test: ## pytest every package
	@for pkg in $(PACKAGES); do echo "==> $$pkg"; $(MAKE) -C $$pkg test; done

serve: ## run the full API + MinIO stack locally (docker compose)
	cd api && docker compose up --build
