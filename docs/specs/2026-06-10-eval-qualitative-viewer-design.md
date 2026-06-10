# Eval qualitative viewer

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan.

## Goal

Add a **qualitative inspection layer** to the `eval/` package: a local, read-only
Streamlit viewer that lets you eyeball the packaged model's behaviour per sequence —
frames with YOLO bboxes overlaid, the extracted smoke tubes, the stabilized tube
crops the classifier sees, and the keep/discard decision — the way
`vision-rd/experiments/temporal-models/temporal-model-explorer` did. It complements
the existing quantitative reports (metrics + PR/ROC/confusion plots), which tell you
*how well* the model does but not *where and why* it succeeds or fails.

Two further outcomes, both deliberate:

1. **Pyro-annotator becomes a first-class eval source.** Eval re-scores the
   human-labeled pyro-annotator sequences with its own packaged `model.zip`,
   producing both metrics and the qualitative view for that set.
2. **A frontend-agnostic data contract.** The viewer reads only generated artifacts
   (`results.json` / `results.parquet` + `details/<key>.json` + frames). A later
   React/Next.js + Tailwind rewrite (its own spec — see Non-goals) consumes the same
   contract with zero rework on the eval/scoring side.

## Background

`eval/` today runs a packaged `model.zip` over directory-convention sequences and
writes `metrics.json`, `predictions.json`, `dropped.json`, and PR/ROC/confusion PNGs
under `data/08_reporting/{split}/vit_dinov2_finetune/`. Ground truth comes from the
directory convention (`wildfire/` → smoke, else fp). Each `SequenceRecord` already
holds `details` — the full passthrough of `output.details` (the
`BboxTubeDetails` schema: `preprocessing.padded_frame_indices`,
`tubes.kept[].entries[]` with bbox/confidence, `decision`). So eval already *captures*
nearly everything the viewer needs; it just doesn't *surface* it per sequence.

The explorer (in the `vision-rd` repo) is a Streamlit app over a "frontend-agnostic
data layer": it reads `results.parquet` + `details/<model>/<key>.json` + a sequence
store, and renders an error-coloured, filterable table plus a drill-down with an
autoplay frame viewer (bboxes + trigger markers), a tube timeline, and context-cropped
tube clips. It never runs the model. Its sequence store carries platform metadata
(org/camera/timestamps), populated for two sources: the platform alert API and a
human-labeled **pyro-annotator** export.

Stabilization is already the production crop mode in this repo
(`docs/specs/2026-06-08-stabilized-crops-production-design.md`): `stabilize=True`
everywhere, and `inference.py` computes a single fixed `window = tube_window(boxes)`
per tube (the union/enclosing box of the tube's observed detections) and crops every
frame from it. That window is **not currently persisted** in `details`.

## Key decisions

These were settled during brainstorming:

- **Interactive viewer, not static artifacts.** A Streamlit app, ported from the
  explorer — for live eyeballing.
- **Adaptive navigation.** Directory-convention train/val sequences carry only
  split + label, so they navigate by split + correctness. Sources that carry richer
  metadata (pyro-annotator) additionally unlock org/camera filtering. The viewer
  shows whatever metadata is present and degrades gracefully when it is absent.
- **Pyro-annotator via copy, not re-import.** We copy the explorer's already-enriched
  sequences (frames + `meta.json`, with org/camera/timestamps) into eval. We do **not**
  port the admin-API import pipeline; no credentials in eval.
- **Eval re-scores the copied sequences.** They run through eval's own packaged
  `model.zip` like train/val, producing eval's own metrics + per-sequence details —
  so the displayed predictions come from the exact model eval evaluates.
- **Persist the stabilized window (Approach B).** Add `stabilized_window` to the
  `KeptTube` details schema and emit the window already computed in `inference.py`.
  The viewer crops every active frame of a tube from this fixed window — the exact
  region the classifier saw.
- **Stabilized is the only crop mode in the UI.** Per the production decision,
  stabilization is always on. The viewer shows the stabilized window as the single
  tube-crop mode: no per-frame/bbox-tracking fallback, no "stabilized" badge, no
  toggle.
- **Frontend-agnostic contract, emit JSON too.** Alongside `results.parquet`, emit
  `results.json` so a future JS frontend reads the table natively.

## Non-goals

- **The React/Next.js + Tailwind rewrite is out of scope here.** It is a separate
  follow-on spec→plan→build cycle that consumes the data contract this spec defines.
  This spec only ensures that contract is clean and documented.
- **No platform API / admin import in eval.** Pyro-annotator data arrives by copy.
- **No multi-model comparison.** Eval is single-variant (`vit_dinov2_finetune`); the
  viewer inspects the one packaged model. (The contract does not preclude adding this
  later, but the UI does not.)
- **No new metrics.** The qualitative layer adds inspection, not new aggregate numbers.

## Architecture

### Data flow

```
explorer store (enriched) ──copy──▶ eval data/01_raw/pyro-annotator/<org>/<camera>/<seq>/{meta.json,images/}
                                                          │
directory-convention  data/01_raw/datasets/{train,val}/{fp,wildfire}/<seq>/images/
                                                          │
                                          evaluate.py (per source) ── model.predict
                                                          │
                       ┌──────────────────────────────────┼───────────────────────────────┐
                       ▼                                   ▼                                ▼
        metrics/predictions/dropped/plots     details/<key>.json (full details,     results.{json,parquet}
                       (unchanged)             incl. stabilized_window)              (one row per sequence)
                                                          │
                                                          ▼
                                  Streamlit viewer (read-only: results + details + frames)
```

### Components

**1. Sequence sources (`store.py`, ported)**
Port the explorer's `store.py`, swapping `from pyrocore import Frame` →
`from temporal_model.core.protocol import Frame`. It reads `meta.json`-based stores:
`SequenceMeta` (key, source, label, org/camera/started_at, frames) + `read_meta` +
`iter_sequence_dirs` + a `build_frames` helper that yields `core.protocol.Frame`s.

`evaluate.py` learns two source kinds behind a common iterator yielding
`(key, frames, label, meta | None)`:
- **directory-convention** (existing): label from path (`is_wf_sequence`), `meta=None`.
- **meta-store** (pyro-annotator): label from `meta.label`; `unknown`-labeled
  sequences are excluded from metrics but retained for the viewer (GT-unknown).

**2. `core` change — persist the stabilized window**
- `details_schema.py`: add `stabilized_window: tuple[float, float, float, float] | None`
  to `KeptTube`.
- `inference.py`: where `window = tube_window(boxes)` is already computed per tube
  (~line 258), carry it into the emitted kept-tube detail. `None` when the tube has no
  usable detection (the existing no-window case) or when `stabilize` is off.

**3. `evaluate.py` emit changes**
In addition to today's outputs, write:
- `details/<key>.json` per sequence — the full `output.details` (now including
  `stabilized_window`). One file per sequence, keyed by sequence key.
- `results.parquet` **and** `results.json` — one row per scored sequence:
  `key, source, split, label, decision, outcome, score, probability,
  trigger_frame_index`, plus `organization_name, camera_name, started_at` when the
  source provides them (else null). `outcome` is the correctness label
  (`kept-smoke` / `discarded-fp` / `discarded-smoke` / `kept-fp` / `n/a`).

These live under the existing reporting tree so the viewer has a single root to read.

**4. Viewer (`render.py` + `app.py`, ported and split)**
Port the explorer's app, split for isolation/testability (the explorer crammed ~670
lines into one file):
- `render.py` — **pure** helpers, no Streamlit: `draw_bboxes`, tube crop (via
  `temporal_model.core.crop`: `expand_bbox`, `norm_bbox_to_pixel_square`,
  `crop_and_resize`), tube-timeline dataframe, `processed_to_input_index` mapping,
  trigger-state logic, correctness labels/colours.
- `app.py` — the Streamlit UI: sidebar source selector; adaptive org/camera filters;
  correctness-coloured, filterable sequence table with score/probability columns; and
  the autoplay drill-down (frame viewer with bboxes + decisive/would-trigger markers,
  tube timeline, and the single-mode **stabilized-window** tube crops). Read-only over
  the emitted artifacts; never runs the model.

**5. Data copy**
A `scripts/` helper (and/or `make` target) performs the one-time copy of the
explorer's `data/03_primary/sequences/pyro-annotator/**` into
`data/01_raw/pyro-annotator/`, DVC-tracked in eval's remote so it travels via
`dvc pull`. `dvc.yaml` gains a `foreach` entry to score this source.

### Data contract (frontend-agnostic interface)

The viewer — and any future frontend — depends only on:
- `results.json` (and `.parquet`): the per-sequence row schema above.
- `details/<key>.json`: the `BboxTubeDetails` shape + `stabilized_window`.
- the sequence frames + `meta.json` (for image paths and metadata).

This contract is documented in the eval README and is the stable boundary the
React/Next.js rewrite will build against.

## Error handling

- Existing strict policy is unchanged: a per-sequence inference exception aborts the
  run; sequences with no images are recorded in `dropped.json` and skipped.
- The viewer is defensive (it reads possibly-partial artifacts): a missing
  `details/<key>.json` shows an empty detail panel; a sequence with no kept tubes
  shows "no smoke tubes extracted"; absent metadata hides the corresponding filters
  rather than erroring.
- `unknown`-labeled pyro-annotator sequences: excluded from metrics, shown with
  GT-unknown colouring in the viewer.

## Testing

- **Pure render helpers** (`render.py`): port the explorer's app-helper tests —
  `processed_to_input_index` (including synthetic/padded slots), trigger-state logic,
  correctness labels, tube-timeline dataframe shape.
- **Source iteration / emit** (`evaluate.py`, `store.py`): a meta-store fixture yields
  the expected `(key, frames, label, meta)`; a scored run writes `details/<key>.json`
  and `results.{json,parquet}` rows matching the documented schema; `unknown` labels
  are excluded from metrics but present in `results`.
- **`core` change**: a kept tube carries the expected `stabilized_window`; a tube with
  no usable detection (or `stabilize=False`) carries `None`. Existing details-schema
  and parity tests still pass.
- Streamlit UI code (`app.py`) stays thin and is excluded from coverage
  (`# pragma: no cover`), as in the explorer.

## Open implementation details (for the plan, not blockers)

- Exact on-disk root for `details/` and `results.*` within `data/08_reporting/`
  (per-split vs combined), chosen to give the viewer one read root.
- Whether the copy helper uses `dvc import`/`dvc get` from the explorer remote or a
  filesystem copy + `dvc add`.
- Pagination/lazy frame loading if a copied source is large (the explorer loaded
  frames per drill-down, which should suffice).
