# Stabilized crops in production train + eval

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan.

## Goal

Bring the experiment's `stabilize` crop feature into the `temporal-model` monorepo
and make **stabilized crops the production crop mode** for the single
`vit_dinov2_finetune` variant. After this change, `dvc repro` produces a stabilized
model and the eval reports stabilized metrics.

## Background

The temporal head is fed per-tube crops. Today each crop recenters and rescales on
**each frame's own bbox** (per-frame mode): the smoke stays centered/same-size while
the background slides — a "jumpy" sequence in which the smoke never appears to move.

The experiment (`vision-rd/experiments/temporal-models/bbox-tube-temporal`, lib
`vision-rd/lib/bbox-tube-temporal`) prototyped and validated an alternative: a
**single fixed crop window per tube** — the union (enclosing) box of the tube's
observed detections — applied to every frame, so the background is static and the
smoke visibly moves/grows. The opt-in `stabilize` flag was threaded through both crop
paths there. A model retrained on stabilized crops is the artifact this work brings
into production.

The production repo is intentionally **single-variant** (ViT-only scope): it packages
exactly one `vit_dinov2_finetune` `model.zip` and evaluates it. We therefore do **not**
mirror the experiment's coexisting-variant stage design. Instead we **replace** the
per-frame crop mode with stabilized; the per-frame baseline survives only in git
history.

## Key decisions

- **Replace, not coexist.** Stage names, data dirs, and the variant name
  (`vit_dinov2_finetune`) are unchanged. The global `model_input.stabilize` param is
  the single source of truth.
- **Stabilized is the default everywhere.** Function-signature defaults, the
  packaged-config fallback, the CLI arg default, and the committed param are all
  `true`. The existing per-frame `model.zip` can be discarded — `dvc repro`
  regenerates the DVC-tracked artifact fresh as stabilized.
- **Surgical port into `core`, not a file copy.** `core/temporal_model/core` mirrors
  the experiment lib but carries repo-specific additions (`aggregation` /
  `logistic_threshold` properties on `model.py`). A wholesale copy would clobber
  those, so we apply only the small stabilize diffs.
- **Code only; re-train here.** No artifact import. `dvc repro` trains a fresh
  stabilized checkpoint in this repo (the repo's existing fresh-train pattern; the
  calibrator may drift a few % from the experiment, as already documented).

## Architecture

Data flow (unchanged shape; the only difference is per-frame box → fixed union box):

```
tubes JSON ── build_model_input (--stabilize true) ──▶ stabilized 224 PNG patches ──▶ train ──▶ package ──▶ model.zip { model_input.stabilize: true }
                          │                                                                                        │
                    union_window (enclosing box of observed detections)                                          ▼
                                                                                          eval: model.predict ── crop_tube_patches(stabilize) ── inference parity
```

`stabilize` is a **crop-window decision**, not a tube-building step. It is threaded
through the two crop functions and baked into the packaged config so a
stabilized-trained model crops consistently when deployed.

## Components

### `core/` — the capability

- **New `stabilize.py`** — port `union_window(boxes) -> (cx, cy, w, h)` verbatim from
  the experiment lib. Pure geometry, no I/O (~25 lines): the enclosing box of a list
  of normalized `(cx, cy, w, h)` boxes; raises `ValueError` on empty input.
- **`model_input.py`** — `process_tube(..., stabilize: bool = True)`. When stabilized,
  compute the union window once from the tube's observed (non-gap) entries — falling
  back to all entries if none are observed — and crop that fixed box
  (`expand_bbox(window, context_factor)` → square → resize) for every frame. The
  `meta.json` records `stabilize`. `stabilize=False` is byte-identical to today's
  per-frame output.
- **`inference.py`** — `crop_tube_patches(..., stabilize: bool = True)`. The same
  switch on the inference path: union of the tube's observed real detections (falling
  back to all detections if none are observed).
- **`model.py`** — `predict` passes `stabilize=mi.get("stabilize", True)` to
  `crop_tube_patches`. A config missing the key now means stabilized; safe because
  `package.py` always bakes an explicit `stabilize` key, so this fallback only ever
  affects hand-written configs.

### `train/` — produce the stabilized model

- **`build_model_input.py`** — add a `--stabilize` CLI arg (default `"true"`) parsed
  by a `_to_bool` helper (mirroring the experiment), threaded into `process_tube`
  alongside the existing `--context-factor` / `--patch-size`.
- **`package.py`** — in `_build_config`, add
  `"stabilize": all_params["model_input"].get("stabilize", True)` to the `model_input`
  dict (line ~104). No new flag — it reads the param. This bakes `stabilize: true`
  into `model.zip` for inference parity. The calibrator is fit on the already
  stabilized `data/05_model_input/val` crops automatically (the `--val-patches-dir`
  default is unchanged).
- **`params.yaml`** — add `stabilize: true` under `model_input`.
- **`dvc.yaml`**:
  - `build_model_input` `do.cmd` gains `--stabilize ${model_input.stabilize}`; add
    `../core/src/temporal_model/core/stabilize.py` to its `deps`.
  - `package` `deps` gains `../core/src/temporal_model/core/stabilize.py` (imported
    transitively via `inference.py` / `model.py`).
  - Both stages already list the whole `model_input` block under `params:`, so the
    new key is auto-tracked — flipping it invalidates crops and package correctly.

### `eval/` — no behaviour change

`evaluate.py` calls `model.predict`, which crops from the baked config, so a
stabilized `model.zip` evaluates stabilized automatically. **Only change:** add
`../core/src/temporal_model/core/stabilize.py` to the `evaluate` stage's `deps` list
for hash correctness.

## Testing

- **core**
  - `test_stabilize.py`: `union_window` — empty raises, single-box passthrough,
    two-box enclosing.
  - `test_model_input.py`: `process_tube(stabilize=True)` yields a constant crop
    window across frames; `stabilize=False` is unchanged; the default (no arg) is the
    stabilized result.
  - `test_inference_units.py`: `crop_tube_patches` uses the union window when
    stabilized; default (no arg) is stabilized.
- **train**
  - `test_package_builders.py::test_build_config_shape`: the baked `model_input`
    config carries `stabilize` — `true` when the param is set, and the
    `.get(..., True)` default when the param is absent.
  - A small `build_model_input --stabilize` test: the flag (and its `_to_bool`
    default) reaches `process_tube`.
- `make lint` + `make test` green in `core`, `train`, and `eval`.

## Out of scope / downstream

- **Re-recording the eval baseline** is a post-merge `dvc repro` step run separately —
  not part of this code change. `dvc repro` regenerates the DVC-tracked
  `model.zip` and the eval reports; no manual deletion of the old artifact is needed.
- No artifact import (fresh train here).
- No coexisting per-frame variant.
- No tuning of `context_factor` or the window definition (held fixed to isolate the
  per-frame → union change).
