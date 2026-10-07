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
crop** (the same fixed window the classifier used, shown with a little extra
context) synced to the current frame, and the
keep/discard decision. Sequences are listed in an error-coloured, filterable table
(missed smoke / false alarm / smoke kept / fp filtered). The viewer never runs the
model — it only reads generated artifacts.

The left pane selects the **source** (`test` by default); org/camera
filters appear only for sources that carry that metadata.

### React / Next.js viewer

A polished React/Next.js + Tailwind port of the same viewer lives in
[`viewer/`](../viewer) and reads the identical reporting tree (it never runs the
model). It needs Node 22+ and a populated reporting tree — pull the packaged
model and artifacts first (same as the Streamlit app):

```bash
# from eval/: fetch model.zip + raw sequences + reporting tree from the DVC remote
uv run dvc pull
# (or rebuild the reporting tree locally: uv run dvc repro)

cd ../viewer
cp .env.local.example .env.local   # DATA_ROOT — defaults to ../eval
npm install
npm run dev                        # http://localhost:3000
```

`DATA_ROOT` points at this `eval/` package dir; the app derives the reporting tree
(`$DATA_ROOT/data/08_reporting`) and resolves frame paths relative to it. See
[`viewer/README.md`](../viewer/README.md) for the full guide.

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
- `model_config.json` — the scored model's merged metadata (detector source +
  variant + train_git_sha from the package manifest; decision/infer/model_input/
  tubes/classifier config; logistic calibrator). Drives the viewer's sidebar
  model-config panel.

This is the stable interface; a future React/Next.js viewer consumes the same files.

## Pipeline

`dvc.yaml` defines one `evaluate` stage on pyro-dataset's `sequential_test`, the
one set models are compared on. It is DVC-imported at the same release as the train
imports (`data/01_raw/sequential_test.dvc`, private remote), and the train workflow
bumps it with them. It includes the former pyro-annotator testbed, now registered
in pyro-dataset, so there is no separate pyro-annotator source any more.

The stage consumes a packaged model at `data/06_models/vit_dinov2_finetune/model.zip`.
It is wired in from the train `package` stage by a local `dvc import-url`
(`model.zip.dvc`): refresh it with `make update-model` after re-packaging in train,
or `dvc pull` it from eval's remote. The stage writes `metrics.json`,
`predictions.json`, `dropped.json`, PR/ROC/confusion PNGs and the viewer artifacts
(`results.{json,parquet}`, `details/`, `sequences/`) under
`data/08_reporting/test/vit_dinov2_finetune/`.

`metrics.json` holds the confusion counts plus prevalence-free metrics:
- recall and FPR with Wilson 95% intervals (`recall_ci95`, `fpr_ci95`);
- `fpr_at_recall_95`, a threshold sweep on the sequence score, for ranking models;
- `detected_within_frames`, the share of all smoke alerted within 2, 3 and 5
  frames (2 is the earliest possible), with missed smoke counted as not detected.

Precision, F1 and PR AUC are still computed, but depend on the test set's
smoke/FP ratio, which is arbitrary.

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
