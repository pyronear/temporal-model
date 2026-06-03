# Model Packaging Pipeline (train) — Design

**Date:** 2026-06-03
**Status:** Implemented (train `package` DVC stage — PR #8)
**Scope:** A `train` DVC stage that turns a trained checkpoint into a deployable,
**calibrated**, version-stamped `model.zip` — bundling the classifier, the
DVC-tracked detector, a built inference `config.yaml`, and a freshly-fit logistic
calibrator + decision threshold. This is the producer of the artifact that the
[API release](2026-06-03-api-release-design.md) bakes into the image; `publish`
to HuggingFace stays a separate step. Builds on
[model-versioning](2026-06-03-model-versioning-design.md).

## Goal

`dvc repro package` produces `train/data/06_models/vit_dinov2_finetune/model.zip`,
a self-describing package: a `provenance` block (train SHA, backbone, detector),
`classifier.ckpt`, `yolo_weights.pt`, `config.yaml`, and `logistic_calibrator.json`.
The human `model_version` is **not** set here — it is applied later by the release
`publish` step (the git tag is the single source of truth). The package stage is
the missing link between `train` (checkpoint) and the release pipeline (publish →
HuggingFace → baked image).

## Background

This is a **port** of the proven vision-rd implementation
(`experiments/temporal-models/bbox-tube-temporal`), adapted to this repo. The key
references:

- `scripts/package_model.py` — the orchestrator (loads classifier, calibrates,
  builds config, calls `build_model_package`).
- `src/bbox_tube_temporal_exp/{calibration,logistic_calibrator_fit,val_predict,package_predict}.py`
  — the calibration/threshold/inference helpers.
- `params.yaml::package` — the inference-config source.

**What already exists here** (no porting needed): `core` has `LogisticCalibrator`,
`extract_features`, `FEATURE_NAMES`, `BboxTubeTemporalModel(yolo, classifier,
config)`, `build_model_package(..., model_version, train_git_sha, calibrator)`,
and `train` has `TubePatchDataset` (`dataset.py`) and `best_checkpoint.pt` (a
Lightning ckpt wrapping `TemporalSmokeClassifier`).

**What's missing** (this spec adds): the `package` DVC stage, the four ported
helper modules, the `package` params section, and `sklearn` as a `train` dep.

**Key insight from vision-rd:** calibration is done **in memory inside the package
script** — it builds a `BboxTubeTemporalModel` from the loose classifier + detector
and runs inference on the raw train/val sequences to fit the calibrator. There is
**no two-pass DVC graph** and **no dependency on the `eval` package**; the package
stage is self-contained.

## The flow (one self-contained `package` stage)

```
best_checkpoint.pt ─┐
yolo_weights.pt ─────┤
params.yaml[package]─┼─►  package.py  ──►  model.zip
val patches ─────────┤      │
raw train/val seqs ──┘      ├─ load classifier from ckpt
                            ├─ collect_val_probabilities → calibrate_threshold(target_recall)  [max_logit thresh]
                            ├─ if aggregation == logistic:
                            │    build in-mem BboxTubeTemporalModel(yolo, classifier, cfg)
                            │    collect_pipeline_records(train) → fit() → LogisticCalibrator
                            │    collect_pipeline_records(val)   → calibrated probs → calibrate_threshold → logistic_threshold
                            ├─ build config.yaml from params.yaml
                            └─ build_model_package(ckpt, yolo, config, calibrator, train_git_sha)  [no model_version]
```

`eval` then consumes the `model.zip` (its `data/06_models/.../model.zip.dvc` is a
local `dvc import-url` of this output — `train` and `eval` are separate DVC
projects, so the hand-off is explicit; refresh with `make -C eval update-model`);
the release `publish` step uploads it to HuggingFace.

## Decisions

| Decision | Choice |
|---|---|
| Stage shape | **One** `package` DVC stage; calibration in-memory (port vision-rd). No two-pass graph, no eval dependency. |
| Scope | **ViT-only** (`vit_dinov2_finetune`) — single stage, no `foreach` over variants. |
| `config.yaml` source | **Built from `params.yaml`** (`package` section + shared `tubes`/`build_tubes`/`model_input`/`train_<variant>` blocks). |
| Calibrator | **Fit in the stage** (`aggregation: logistic` for ViT): sklearn `LogisticRegression` on pipeline records, numpy/sklearn parity-checked, with sanity checks. |
| Threshold | `calibrate_threshold` picks the smallest prob threshold meeting `target_recall` (0.95) on val. |
| Detector | From the DVC-tracked `train/data/06_models/detectors/<name>/yolo_weights.pt` (not vision-rd's `data/01_raw/models/best.pt`). |
| Version | **Not set here.** The package stage stamps `provenance` (incl. `train_git_sha` from `git rev-parse HEAD`) but leaves `model_version` unset — the release `publish` step applies it from the git tag. Keeps a release label out of the hyperparameter file and out of DVC's re-run graph. |
| Publish | **Separate** — the stage only outputs `model.zip` (DVC-tracked); the release `publish` step uploads to HuggingFace. |
| Module home | `temporal_model.train` (mirrors vision-rd's exp-package placement; `core` keeps the primitives). |

## Components

### Ported helper modules (`temporal_model.train`)

| Module | Responsibility | Notes |
|---|---|---|
| `calibration.py` | `calibrate_threshold(probs, labels, *, target_recall) -> float` | Pure numpy; smallest threshold meeting recall. |
| `logistic_calibrator_fit.py` | `fit(records) -> LogisticCalibrator` | Imports **sklearn**; fits on the 4 features (`logit/log_len/mean_conf/n_tubes`), numpy↔sklearn parity check (`atol=1e-6`), 3 spanning sanity checks. |
| `val_predict.py` | `collect_val_probabilities(classifier, val_patches_dir, *, max_frames, …)` | Runs the classifier over `TubePatchDataset` → `(probs, labels)`. |
| `package_predict.py` | `collect_pipeline_records(*, model, raw_dir) -> list[dict]` | Runs the full in-mem pipeline per labelled sequence → `{label, sequence, kept_tubes}`. |

### `package.py` — the orchestrator CLI (`python -m temporal_model.train.package`)

Ports `scripts/package_model.py`:
1. Load all params; read `train_<variant>` + `package` blocks.
2. `_load_classifier_from_ckpt(best_checkpoint.pt, variant_cfg)` (strip `model.` prefix).
3. `collect_val_probabilities` → `calibrate_threshold(target_recall)` → `threshold`.
4. If `aggregation == "logistic"`: build a pipeline config (`max_logit`), construct an
   in-memory `BboxTubeTemporalModel`, `collect_pipeline_records(train)` → `fit()` →
   calibrator; `collect_pipeline_records(val)` → calibrated probs →
   `calibrate_threshold` → `logistic_threshold`.
5. `_build_config(...)` → the packaged `config.yaml` dict.
6. `build_model_package(yolo_weights_path, classifier_ckpt_path, config, variant,
   output_path, calibrator, train_git_sha=<git rev-parse HEAD>)` — **no
   `model_version`** (applied later by `publish`).

### `params.yaml` — `package` section (ported)

```yaml
package:
  target_recall: 0.95
  infer_min_tube_length: 2
  infer:
    confidence_threshold: 0.1
    iou_nms: 0.2
    image_size: 1024
    pad_to_min_frames: 20
    pad_strategy: symmetric
  aggregation:
    vit_dinov2_finetune: logistic
```

`config.yaml` is assembled from this plus the existing `tubes`, `build_tubes`,
`model_input`, and `train_vit_dinov2_finetune` blocks (single source of truth —
shared values like tube thresholds never drift).

### `dvc.yaml` — `package` stage

```yaml
package:
  cmd: >-
    uv run python -m temporal_model.train.package
    --variant vit_dinov2_finetune
    --output data/06_models/vit_dinov2_finetune/model.zip
  deps:
    - src/temporal_model/train/package.py
    - src/temporal_model/train/calibration.py
    - src/temporal_model/train/logistic_calibrator_fit.py
    - src/temporal_model/train/val_predict.py
    - src/temporal_model/train/package_predict.py
    - src/temporal_model/train/dataset.py
    - ../core/src/temporal_model/core/package.py
    - ../core/src/temporal_model/core/model.py
    - ../core/src/temporal_model/core/inference.py
    - ../core/src/temporal_model/core/logistic_calibrator.py
    - ../core/src/temporal_model/core/tubes.py
    - ../core/src/temporal_model/core/data.py
    - data/06_models/vit_dinov2_finetune/best_checkpoint.pt
    - data/06_models/detectors/yolo11s_nimble-narwhal_v6.0.0/yolo_weights.pt
    - data/05_model_input/val
    - data/01_raw/datasets/train
    - data/01_raw/datasets/val
  params:
    - package
    - tubes
    - build_tubes
    - model_input
    - train_vit_dinov2_finetune
  outs:
    - data/06_models/vit_dinov2_finetune/model.zip
```

(No `model_version` here — it isn't a hyperparameter and shouldn't gate the
stage's re-run. `train_git_sha` is stamped at run time and is not a dep — it
reflects the commit at the last `dvc repro`. The human version is applied later by
`publish`.)

## Dependencies

- Add `scikit-learn` to `train`'s `pyproject.toml` (only `logistic_calibrator_fit`
  imports it; the runtime inference path stays sklearn-free). Re-lock `train`.

## Porting-verification notes

Confirm during implementation (vision-rd's lib API may differ slightly from our
migrated `core`):

- **Model method names:** `package_predict` calls `model.load_sequence(frame_paths)`
  and `model.predict(frames)` with `out.details["tubes"]["kept"]`. Our `core.model`
  exposes `predict_sequence`; verify/expose the `load_sequence` + `predict`
  decomposition (or adapt `collect_pipeline_records` to `predict_sequence`).
- **`core.data` helpers:** `package_predict` imports `get_sorted_frames`,
  `is_wf_sequence`, `list_sequences` — confirm these exist in our `core.data`.
- **Dataset batch keys:** `val_predict` expects `batch["patches"|"mask"|"label"]`
  from `TubePatchDataset` — confirm our `train/dataset.py` yields these.
- **`classifier(patches, mask)`** forward signature matches.

Any gap becomes a small adapter, not a redesign.

## Testing

- `calibration.calibrate_threshold` — unit tests: exact recall boundary, all-recall
  (`target_recall=1.0`), no-positives raises, mis-shaped raises (port vision-rd's).
- `logistic_calibrator_fit.fit` — fits on synthetic records, asserts parity check
  passes and sanity checks round-trip through `verify_sanity_checks`; single-class
  raises.
- `val_predict` / `package_predict` — small fixtures (a couple of fake sequences /
  a tiny patches dir), assert record/prob shapes; mock YOLO where needed (as
  `core` tests already do via `_load_yolo`).
- `package.py` `_build_config` / `_classifier_kwargs` / `_tubes_config` — pure-dict
  unit tests against a sample `params.yaml`.
- End-to-end `package` run is exercised manually / by `dvc repro` (needs the real
  checkpoint + detector + data); not in fast CI.

## Non-goals

- Multi-variant packaging (ViT-only repo); the `aggregation` map keeps one entry.
- A `max_logit`-only path beyond what the port carries (kept for completeness but
  ViT uses `logistic`).
- Publishing to HuggingFace (separate `publish` step, API release spec).
- The `analyze_variant` research/reporting stage (vision-rd's `recommended_config`
  + analysis report) — not needed to produce a deployable package.
- Re-deriving the detector here (it's DVC-tracked already).

## Success criteria

1. `dvc repro package` produces `data/06_models/vit_dinov2_finetune/model.zip` with
   a calibrated `logistic_calibrator.json`, a built `config.yaml`, and a manifest
   carrying a `provenance` block (train SHA, backbone, detector) and **no**
   `model_version` (applied later by the release `publish` step).
2. `config.yaml` is assembled from `params.yaml` (no hand-authored drift on shared
   fields).
3. The fitted calibrator passes its numpy/sklearn parity check and its embedded
   sanity checks `verify_sanity_checks()` at load.
4. `decision.threshold` achieves ≥ `target_recall` (0.95) on val.
5. The produced `model.zip` loads via `core.load_model_package` and runs an
   end-to-end `predict` (the existing integration test path).
6. Unit tests cover `calibrate_threshold`, `fit`, and the config builders without
   GPU/network.
7. The artifact feeds the release pipeline unchanged: `publish --version <v>` then
   tag `v<v>` bakes it into the image.
