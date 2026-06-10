# Figure generators for [docs/README.md](../../README.md)

Each script regenerates one or more figures in `docs/assets/`, running the
real pipeline from `core` with the released model on DVC-tracked training
data.

## Prerequisites

```bash
make fetch-model              # api/models/model.zip
cd train && make install && dvc pull   # deps + raw sequences + patch crops
```

## Run

From `train/` (its venv has `core`, torch, and matplotlib):

```bash
uv run python ../docs/assets/scripts/<script>.py
```

| Script | Figures |
|---|---|
| `make_motivation_and_step4.py` | `avinyonet_annotated/patches`, `brison_frame_strip/annotated/patches` |
| `make_unstabilized_strip.py` | `brison_patches_unstabilized` |
| `make_tube_timeline.py` | `brison020_tube_stages` |
| `make_gap_crops.py` | `brison020_gap_crops` |
| `make_stabilized_strip.py` | `brison020_patches_stabilized` |
| `make_step5_inputs.py` | `brison_backbone_input`, `brison_head_input` |
| `make_step6_calibrator.py` | `calibrator_curves` |
| `scan_sequences.py` | (no figure) ranks sequences where filter/merge/interpolate visibly act, for picking illustration candidates |

Figures embed model outputs (logits, thresholds, calibrator coefficients), so
regenerating against a different released `model.zip` will change them —
re-stamp the version mentions in `docs/README.md` if you do.
