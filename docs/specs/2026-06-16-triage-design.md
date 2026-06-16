# triage/: Backlog Triage of the Annotation Queue

**Date:** 2026-06-16
**Status:** Draft

## Motivation

The pyro-annotator (https://annotator.pyronear.org) accumulates a large backlog
of **unannotated** sequences — alerts imported from the source APIs that no
human has labelled yet. Most are false positives, so a person grinding through
the whole queue spends most of their time discarding obvious non-smoke.

`triage/` shrinks that queue. It pulls the unannotated backlog (read-only),
scores every sequence with the temporal smoke classifier in-process, and splits
the backlog at a configurable probability threshold:

- **low (`< threshold`)** — almost certainly not smoke. Emitted as a read-only
  **worklist** the team can later apply to the annotator as the `unlabeled`
  false-positive type (the annotator's own concept for "false positive
  discarded by auto-annotation without category").
- **high (`>= threshold`)** — worth a human's eyes. Written into the eval-viewer
  contract so they are reviewed locally in `viewer/`.

The default threshold is **0.35**.

## Scope & constraints

- **API host.** The REST API is `annotationapi.pyronear.org` (the live FastAPI
  service); `annotator.pyronear.org` is the frontend SPA, not the API.
- **Read-only against the pyro-annotator API.** `triage` only issues `GET`s
  (plus the one `POST /api/v1/auth/login` to obtain a bearer token). It never
  writes annotations to production. The low-score "assign to unlabeled" action is
  emitted as an artifact (sequence ids + a ready-to-POST `bulk` payload) and
  applied by a human in a separate, deliberate step — out of scope for this
  package.
- **In-process scoring via `core`, no Docker.** We score for triage, not to
  audit production parity, so we call `BboxTubeTemporalModel.predict()`
  directly (like `benchmark`) rather than replaying through the pinned API
  image (like `monitor`).
- **Test on small subsets first.** `pull` supports `--limit`, and the intended
  workflow ramps `--limit 3` → `--limit ~50` → full pull.

## Decisions (agreed in brainstorming)

1. **New sibling package `triage/`** (`temporal_model.triage`, script
   `temporal-triage`, distribution `temporal-model-triage`), depending on
   `core` via the `uv` path source — same shape as `benchmark`/`monitor`.
2. **Pull scope = `processing_stage=ready_to_annotate`.** The deployed
   annotator pre-creates a `SequenceAnnotation` record at `ready_to_annotate`
   for the whole human queue, so `has_annotation=false` is near-empty (2 live)
   while `ready_to_annotate` is the real backlog (21,489 live, matching the
   UI's "Ready to Annotate" count). The `GET /sequences/` listing is not
   implicitly org-scoped, so an account token pages the global backlog across
   all organizations. The stage is a `--stage` flag (default
   `ready_to_annotate`) so other stages can be pulled later.
3. **Read-only; low scorers become a worklist, not a write.** `triage` emits
   `unlabeled.json` (ids + bulk-unlabel payload). Applying it is a separate
   human step.
4. **In-process `core` scoring.** Sequence score = **max `probability` over
   kept tubes** from `predict()`'s `details`. Sequences with no kept tubes (or
   an uncalibrated model) score `0.0` and fall in the low bucket.
5. **Threshold is a DVC param.** `triage/params.yaml` holds
   `triage.threshold: 0.35`, listed under the `score` stage's `params:` so a
   change reruns the stage. A `--threshold` CLI flag overrides it for ad-hoc
   runs.
6. **Reuse the Next.js eval viewer, no new UI.** The high bucket is written in
   the exact reporting contract `viewer/` already reads
   (`results.json`, `details/<key>.json`, `sequences/<key>.json`,
   `model_config.json` under `data/08_reporting/<source>/vit_dinov2_finetune/`),
   browsed with `DATA_ROOT=../triage`.

## Architecture

Two CLI commands.

### `temporal-triage pull` (read-only)

1. `POST /api/v1/auth/login` → bearer token (creds via `.envrc`, like
   `monitor`).
2. Page `GET /api/v1/sequences/?processing_stage=ready_to_annotate` (newest
   first), honouring `--stage` / `--limit` / `--page-size`.
3. For each sequence not already on disk: `GET /api/v1/detections/?sequence_id=…`
   for its frames, `GET /api/v1/detections/{id}/url` for each signed image URL,
   download the image, and write the store entry.
4. Incremental — sequences already present are skipped.

Store layout (mirrors `monitor`/`benchmark`):

```
data/01_raw/sequences/<org>/<camera>/seq_<id>/
    meta.json        # SequenceMeta: key, ids, org/camera, ordered frames
    images/<frame>.jpg
```

`meta.json` carries the viewer join key (`pyro-annotator_<sequence_id>`), the
ordered frame list (`file`, `detection_id`, `recorded_at`, `bucket_key`), and
provenance. There is **no ground-truth label** (these are unannotated) — the
store's `label` is `"unknown"`.

Exposed as `make pull` → runs `pull`, then `dvc add data/01_raw/sequences` +
`dvc push`, so the pulled store is shareable.

### `temporal-triage score` (DVC stage)

1. Load each stored sequence into `core` `Frame`s (reuse `benchmark`'s
   `dataset.py` loader shape).
2. `model = BboxTubeTemporalModel.from_package(model.zip, device=auto)`;
   `output = model.predict(frames)`.
3. Sequence score = max kept-tube `probability` (`0.0` if none).
4. Split at `threshold` (param, default 0.35) into low / high.
5. Write outputs (below).

`model.zip` is read from `api/models/` (populated by `make fetch-model`, same
as `benchmark`).

### Module layout (one purpose each)

| Module | Purpose |
|--------|---------|
| `annotator_api.py` | Read-only client: `login`, `iter_unannotated_sequences`, `list_detections`, `detection_image_url`. |
| `pull.py` | Orchestrate the pull; build `SequenceMeta`, download frames into the store. |
| `store.py` | `meta.json` schema (`SequenceMeta`/`FrameMeta`) + read/write/iter. |
| `score.py` | Load store → `predict` → sequence score; classify low/high. |
| `report.py` | Write eval-viewer contract for the high bucket + `unlabeled.json` / `review.json` worklists. |
| `cli.py` | `pull` / `score` subcommands. |

## Outputs

```
data/08_reporting/pyro-annotator/vit_dinov2_finetune/
    results.json            # eval-viewer rows (HIGH bucket): key, score, decision
    details/<key>.json      # tubes/boxes/crops for the viewer (HIGH bucket)
    sequences/<key>.json    # frame refs for the viewer
    model_config.json       # model provenance
    unlabeled.json          # LOW (<threshold): sequence_ids, scores, ready-to-POST bulk-unlabel payload
    review.json             # HIGH (>=threshold): sequence_ids, scores
    dropped.json            # skip reasons (no_images, predict_failed, ...)
```

`unlabeled.json` records, per sequence: `sequence_id`, `key`, `score`, plus a
top-level `bulk_payload` block shaped for `POST /api/v1/annotations/sequences/bulk`
with `false_positive_type: "unlabeled"` — copy-pasteable for the separate apply
step, never sent by this package.

## Testing

Offline, like `monitor`: mocked HTTP for the client, a tiny on-disk fake store
for `score`/`report`, no network and no Docker. `score`'s model call is
exercised against a minimal packaged model or stubbed `predict` in unit tests;
a real end-to-end run is the `--limit 3` manual smoke test.

## Out of scope

- Writing annotations back to the annotator (the apply step).
- Production-parity scoring (ROI envelope / windowing) — that is `monitor`.
- Per-organization or per-camera filtering beyond what the listing offers
  (can be added as `pull` flags later if needed).
