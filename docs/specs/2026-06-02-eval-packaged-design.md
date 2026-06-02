# eval/ packaged-model evaluation — design

Date: 2026-06-02
Status: approved

## Goal

Replace the `eval/` scaffold stub with a DVC-driven pipeline that evaluates a
**packaged `model.zip`** end-to-end over raw image sequences and emits
leaderboard-schema metrics plus PR/ROC/confusion-matrix plots.

This is a faithful, **ViT-only**, **core-only** port of the `evaluate_packaged`
path from `vision-rd/experiments/temporal-models/bbox-tube-temporal`. It mirrors
the porting convention already used for the merged `train` package: experiment
scripts/modules move into `src/temporal_model/eval/`, are invoked via
`python -m temporal_model.eval.<module>`, and reference `core` at
`../core/src/temporal_model/core/...`.

## Source

From `vision-rd/experiments/temporal-models/bbox-tube-temporal`:

- `scripts/evaluate_packaged.py` — the end-to-end packaged-model evaluator.
- `src/bbox_tube_temporal_exp/protocol_eval.py` — `SequenceRecord`, `build_record`,
  `compute_metrics` (protocol-level metrics over `TemporalModelOutput`).
- `src/bbox_tube_temporal_exp/eval_plots.py` — matplotlib PR/ROC/confusion helpers.

The checkpoint-level `evaluate.py`, `analyze_variant`, `compare_variants`, the
multi-variant DVC stages, and the leaderboard's multi-model registry are
**out of scope** (see Non-goals).

## Modules

Ported file-for-file with import rewrites only (no logic changes):

| New file | Source | Import rewrites |
|---|---|---|
| `src/temporal_model/eval/evaluate.py` | `scripts/evaluate_packaged.py` | `bbox_tube_temporal.data` → `temporal_model.core.data`; `bbox_tube_temporal.model` → `temporal_model.core.model`; `bbox_tube_temporal_exp.{eval_plots,protocol_eval}` → `temporal_model.eval.{eval_plots,protocol_eval}` |
| `src/temporal_model/eval/protocol_eval.py` | `bbox_tube_temporal_exp/protocol_eval.py` | `pyrocore` → `temporal_model.core.protocol` |
| `src/temporal_model/eval/eval_plots.py` | `bbox_tube_temporal_exp/eval_plots.py` | none |

`evaluate.py` exposes `main()`, so the existing
`temporal-eval = temporal_model.eval.evaluate:main` entry point in
`pyproject.toml` stays valid. The stub `evaluate.py` (which only raised
`SystemExit`) and its `__init__.py` "scaffold stub" docstring are replaced.

Verified that `core` already exposes every symbol the port imports:
- `core.data`: `list_sequences`, `is_wf_sequence`, `get_sorted_frames`
- `core.model`: `BboxTubeTemporalModel.from_archive`, `.predict`
- `core.protocol`: `Frame`, `TemporalModelOutput`, `TemporalModel.load_sequence`

## Data flow

```
model.zip ─┐
           ├─> BboxTubeTemporalModel.from_archive
sequences ─┘            │
   (per sequence_dir)   │
   get_sorted_frames ── model.load_sequence ── model.predict ── build_record
                                                                     │
                                          compute_metrics(records) ──┤
                                                                     v
   output-dir/: metrics.json, predictions.json, dropped.json,
                pr_curve.png, roc_curve.png,
                confusion_matrix.png, confusion_matrix_normalized.png
```

Ground truth is the directory convention (`wildfire/` → `smoke`, else `fp`),
via `core.data.is_wf_sequence`. Sequence-level score is `max(tube_logits)`
(the `max_logit` aggregation baked into the packaged config). TTD is read
directly from `output.trigger_frame_index` (frames), not timestamps.

Error policy is preserved from the source: a per-sequence exception aborts the
run; only sequences with no images are recorded in `dropped.json` and skipped.

## DVC pipeline

Replace the `noop` placeholder in `eval/dvc.yaml` with a single `evaluate`
stage, scoped to the one ViT variant and run `foreach: [train, val]`:

```yaml
stages:
  evaluate:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python -m temporal_model.eval.evaluate
        --model-zip data/06_models/vit_dinov2_finetune/model.zip
        --sequences-dir data/01_raw/datasets/${item}
        --output-dir data/08_reporting/${item}/vit_dinov2_finetune
        --model-name vit_dinov2_finetune-${item}
      deps:
        - src/temporal_model/eval/evaluate.py
        - src/temporal_model/eval/protocol_eval.py
        - src/temporal_model/eval/eval_plots.py
        - ../core/src/temporal_model/core/data.py
        - ../core/src/temporal_model/core/model.py
        - ../core/src/temporal_model/core/inference.py
        - ../core/src/temporal_model/core/tubes.py
        - ../core/src/temporal_model/core/model_input.py
        - ../core/src/temporal_model/core/protocol.py
        - data/06_models/vit_dinov2_finetune/model.zip
        - data/01_raw/datasets/${item}
      outs:
        - data/08_reporting/${item}/vit_dinov2_finetune/predictions.json:
            cache: false
        - data/08_reporting/${item}/vit_dinov2_finetune/dropped.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/vit_dinov2_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/vit_dinov2_finetune/pr_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/roc_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix_normalized.png
```

**`model.zip` is an external input dependency.** Nothing in this repo packages
it yet (the `api` package likewise treats `model.zip` as a provided artifact);
the user supplies it under `data/06_models/vit_dinov2_finetune/model.zip` or
pulls it via DVC.

The stage references no params, so the placeholder `eval/params.yaml` is
**removed**.

## Dependencies

`eval/pyproject.toml` currently declares only `temporal-model-core` and
`pyyaml`. Add the direct runtime deps the port uses:

- `numpy`
- `scikit-learn` (PR/ROC AUC, curves)
- `matplotlib` (plots)
- `tqdm` (progress bar)

`pyyaml` is unused by this port (no params file is read) — remove it.

## Tests

Port the three relevant test files with the same import rewrites; they give us
parity coverage without loading a real model:

- `tests/test_protocol_eval.py` (~262 L) — `build_record` / `compute_metrics`
  over synthetic `TemporalModelOutput`s.
- `tests/test_eval_plots.py` (~87 L) — plot helpers write valid PNGs, including
  the single-class / no-positives placeholder paths.
- `tests/test_evaluate_packaged_driver.py` (~228 L) — drives `evaluate.main`
  with a **mocked** `BboxTubeTemporalModel`, asserting the output files and
  schema. Rewrite the monkeypatch target from `bbox_tube_temporal.model` to
  `temporal_model.core.model` (and the script module to
  `temporal_model.eval.evaluate`).

Replace the current `tests/test_smoke.py` import-stub assertions; keep a minimal
import smoke check if it adds value over the driver test.

## CI / README

- CI already runs `eval` in the matrix (`.github/workflows/ci.yml`) — no change.
- Update `eval/README.md` to describe the real pipeline (drop "scaffold stub").
- Update root `README.md`: flip `eval` status from `scaffold` to `implemented`
  once landed.

## Non-goals

- Checkpoint-level `evaluate.py` (`--arch`, `TubePatchDataset`,
  `LitTemporalClassifier`) — would couple `eval` to `train`.
- `analyze_variant`, `compare_variants`, calibrator fitting, packaging.
- Multi-variant (gru/convnext/in21k/frozen) stages — repo is ViT-only.
- The leaderboard's multi-model registry / ranking — needs other model repos
  not present here.
- FiftyOne error-exploration scripts.

## Success criteria

- `cd eval && make install && make test` passes (ported parity tests green).
- `make lint` clean (ruff).
- `python -m temporal_model.eval.evaluate --help` works via the package.
- Given a `model.zip` + a sequences dir, the stage produces `metrics.json`,
  `predictions.json`, `dropped.json`, and the four PNG plots, with
  `metrics.json` fields matching the leaderboard schema
  (`precision/recall/f1/fpr/mean_ttd_frames/median_ttd_frames/pr_auc/roc_auc`).
