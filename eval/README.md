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
  per-sequence predictions, plots, and the viewer artifacts (see below).
- `protocol_eval.py` — `SequenceRecord` + `build_record` + `compute_metrics`
  (precision/recall/F1/FPR, mean/median TTD in frames, PR/ROC AUC). Field names
  and rounding match the leaderboard schema.
- `eval_plots.py` — matplotlib PR/ROC/confusion-matrix helpers.
- `store.py` — reader for `meta.json`-based sequence sources (pyro-annotator).
- `outcomes.py` — pure decision/outcome/correctness helpers (no I/O).
- `view_store.py` — the normalized per-sequence record (`SequenceView`) the
  viewer reads.
- `render.py` — pure frame/tube drawing helpers (bbox overlay, stabilized crop,
  tube timeline); no Streamlit, so they stay unit-tested.
- `app.py` — the read-only Streamlit viewer (`make app`).

## Qualitative viewer

`make app` launches a local, read-only Streamlit viewer over the reporting tree.
For each sequence it shows the frames with YOLO bboxes overlaid (the decisive /
would-trigger tubes flagged), the per-tube timeline, each kept tube's **stabilized
crop** (the fixed window the classifier saw) synced to the current frame, and the
keep/discard decision. Sequences are listed in an error-coloured, filterable table
(missed smoke / false alarm / smoke kept / fp filtered). The viewer never runs the
model — it only reads generated artifacts.

The left pane selects the **source** (train / val / pyro-annotator); org/camera
filters appear only for sources that carry that metadata.

### Data contract (frontend-agnostic)

The viewer — and any future frontend — depends only on these per-source artifacts
under `data/08_reporting/<source>/vit_dinov2_finetune/`:

- `results.json` (and `results.parquet`) — one row per sequence: `key, source,
  label, decision, outcome, score, probability, trigger_frame_index`, plus
  `organization_name, camera_name, started_at` when the source provides them.
- `details/<key>.json` — the full `BboxTubeDetails` (preprocessing, kept tubes
  with per-frame entries, decision) including `stabilized_window` per kept tube.
- `sequences/<key>.json` — `SequenceView`: key, source, label, metadata, and the
  ordered frame paths (relative to the eval package dir).

This is the stable interface; a future React/Next.js viewer consumes the same files.

## Pyro-annotator source

Pyro-annotator sequences (human-labeled smoke/fp/unknown, enriched with
org/camera/timestamps) are a first-class eval source — re-scored by eval's own
`model.zip`, so the displayed predictions come from the exact model eval evaluates.
The sequences are DVC-tracked (`data/01_raw/pyro-annotator.dvc`); pull them, then
run the dedicated stage:

```bash
uv run dvc pull data/01_raw/pyro-annotator.dvc   # frames + meta.json from eval's remote
uv run dvc repro evaluate_pyro_annotator         # score + emit viewer artifacts
```

`unknown`-labeled sequences are excluded from metrics but remain viewable
(ground-truth-unknown colouring), and pyro-annotator is the viewer's default source.

**Provenance.** The store was copied once from the temporal-model-explorer's
processed pyro-annotator sequences (`data/03_primary/sequences/pyro-annotator/` —
each a `meta.json` + `images/`) and `dvc add`ed here. To refresh after the
explorer's annotations change, re-copy those sequence directories into
`data/01_raw/pyro-annotator/` and `dvc add` again.

## Pipeline

`dvc.yaml` defines an `evaluate` stage run `foreach` train/val, plus an
`evaluate_pyro_annotator` stage for the meta-store source. They consume a
packaged model at `data/06_models/vit_dinov2_finetune/model.zip` — wired in from
the train `package` stage via a local `dvc import-url` (`model.zip.dvc`); refresh
it with `make update-model` after re-packaging in train, or `dvc pull` it from
eval's remote — and raw sequences under
`data/01_raw/datasets/{train,val}/{fp,wildfire}/<seq>/images/`,
writing `metrics.json`, `predictions.json`, `dropped.json`, PR/ROC/confusion PNGs,
and the viewer artifacts (`results.{json,parquet}`, `details/`, `sequences/`)
under `data/08_reporting/{source}/vit_dinov2_finetune/`.

Ground truth comes from the directory convention (`wildfire/` → smoke, else fp).
Error policy is strict: any per-sequence inference exception aborts the run;
sequences with no images are recorded in `dropped.json` and skipped.

## Run

```bash
make install
make test
uv run dvc repro            # needs model.zip + raw sequences in place
make app                    # launch the qualitative viewer (reads the reporting tree)
```
