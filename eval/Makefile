.PHONY: install lint format test update-model app

install: ## uv sync
	uv sync

app: ## launch the Streamlit viewer
	uv run streamlit run src/temporal_model/eval/app.py

update-model: ## re-import the packaged model.zip from train (after dvc repro package)
	uv run dvc update data/06_models/vit_dinov2_finetune/model.zip.dvc

lint: ## ruff check
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

test: ## pytest
	uv run pytest tests/ -v
