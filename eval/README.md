# temporal-model-eval

DVC evaluation pipeline for the temporal smoke classifier: runs a packaged
`model.zip` end-to-end over raw image sequences and reports protocol-level
metrics plus PR/ROC/confusion-matrix plots.

Import as `temporal_model.eval`; CLI entry point `temporal-eval` (also runnable
as `python -m temporal_model.eval.evaluate`). Depends on `temporal-model-core`.

## Modules

- `evaluate.py` — the packaged-model evaluator. Loads `model.zip` via
  `core.model.BboxTubeTemporalModel.from_archive`, iterates the sequences in a
  split, calls `load_sequence` + `predict` per sequence, and writes metrics,
  per-sequence predictions, and plots.
- `protocol_eval.py` — `SequenceRecord` + `build_record` + `compute_metrics`
  (precision/recall/F1/FPR, mean/median TTD in frames, PR/ROC AUC). Field names
  and rounding match the leaderboard schema.
- `eval_plots.py` — matplotlib PR/ROC/confusion-matrix helpers.

## Pipeline

`dvc.yaml` defines one `evaluate` stage, run `foreach` train/val. It consumes a
packaged model at `data/06_models/vit_dinov2_finetune/model.zip` (an **external
input** — nothing in this repo builds it; supply it or pull via DVC) and raw
sequences under `data/01_raw/datasets/{train,val}/{fp,wildfire}/<seq>/images/`,
writing `metrics.json`, `predictions.json`, `dropped.json`, and PR/ROC/confusion
PNGs under `data/08_reporting/{split}/vit_dinov2_finetune/`.

Ground truth comes from the directory convention (`wildfire/` → smoke, else fp).
Error policy is strict: any per-sequence inference exception aborts the run;
sequences with no images are recorded in `dropped.json` and skipped.

## Run

```bash
make install
make test
uv run dvc repro            # needs model.zip + raw sequences in place
```
