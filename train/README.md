# temporal-model-train

DVC training pipeline for the temporal smoke classifier. Import as
`temporal_model.train`; CLI entry point `temporal-train`. Depends on
`temporal-model-core`.

## Pipeline

`dvc.yaml` defines the stages (data-prep stages run `foreach` train/val):

1. `truncate` — cap each sequence to `truncate.max_frames` frames.
2. `build_tubes` — greedy-IoU tube linking from the label detections
   (`core.tubes`); no YOLO inference (labels carry the boxes).
3. `build_model_input` — crop each tube to 224×224 PNG patches (`core.model_input`).
4. `train` — train the `vit_dinov2_finetune` model (ViT-DINOv2 backbone +
   transformer head) via PyTorch Lightning; writes `best_checkpoint.pt`,
   metrics, and training-curve plots under `data/06_models/vit_dinov2_finetune/`.

Hyperparameters live in `params.yaml` (`train_vit_dinov2_finetune` section).
Data-prep modules are invoked as `python -m temporal_model.train.<stage>`.

## Run

Expects raw data under `data/01_raw/datasets_full/{train,val}/{fp,wildfire}/<seq>/{images,labels}/`.

```bash
make install
uv run dvc repro            # full pipeline (uses GPU for training when available)
uv run dvc repro train      # just the training stage (data-prep cached)
```
