# This folder is a mirror of the HuggingFace Space, so it is shaped like one
# (app.py + requirements.txt + README.md), not like the repo's uv packages.
SPACE ?= pyronear/Pyronear-Wildfire-Detection

.PHONY: install run lint format push

install: ## build .venv from requirements.txt (pulls torch — large)
	uv venv --python 3.12
	uv pip install --python .venv -r requirements.txt

run: ## serve the demo at http://localhost:7860
	.venv/bin/python app.py

lint: ## ruff check
	uv run --project ../core ruff check .

format: ## ruff format
	uv run --project ../core ruff format .

push: ## mirror this folder to the Space (needs `hf auth login` or HF_TOKEN)
	uvx --from huggingface_hub hf upload $(SPACE) . . \
	    --repo-type space --exclude ".venv/*" --exclude ".cache/*" --exclude "*__pycache__/*" \
	    --delete "*" \
	    --commit-message "Sync from temporal-model/demo"
