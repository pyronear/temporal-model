# 🗂️🔥 triage — annotation-backlog triage

Shrinks the pyro-annotator's **annotation backlog**. It pulls the unannotated
queue (sequences at `processing_stage=ready_to_annotate`) from the API
**read-only**, scores every sequence with the temporal smoke classifier
in-process, and splits the backlog at a threshold (default **0.35**):

- **To Review (`score ≥ 0.35`)** → reviewed locally in the [`viewer/`](../viewer):
  worth a human's eyes.
- **Unlabel (`score < 0.35`)** → `unlabeled.json`: a read-only worklist (sequence
  ids + a ready-to-send `bulk` body) to later mark these as the `unlabeled`
  false-positive type. triage **never writes** to the annotator.

> **API host.** The REST API is **`annotationapi.pyronear.org`** —
> `annotator.pyronear.org` is the frontend SPA, not the API.

Designs: [`docs/specs/2026-06-16-triage-design.md`](../docs/specs/2026-06-16-triage-design.md)
· [sharding](../docs/specs/2026-06-16-triage-sharding-design.md)

## Read-only guarantee

The only non-GET request triage sends to the annotator is the login POST that
mints a bearer token. The HTTP client has **no** patch/put/delete/non-login-post
method (enforced by `tests/test_annotator_api.py::test_client_is_read_only_only_login_posts`).
The `bulk` payload in `unlabeled.json` is written to disk only — applying it is a
separate, deliberate human step.

---

## Just visualizing? Pull the shared data (no annotator creds, no model, no Docker)

The scored store + report are pushed to the DVC S3 remote, so browsing what the
model decided needs only S3 read access (`AWS_PROFILE=pyronear`) — no annotator
credentials, no `model.zip`, no Docker:

```bash
cd triage
make install                                   # uv sync (brings dvc[s3])
AWS_PROFILE=pyronear uv run dvc pull            # fetch store + report from S3

cd ../viewer
npm install                                    # first time only
DATA_ROOT=../triage npm run dev                # browse at http://localhost:3000
```

`dvc pull` fetches **both** trees in one go: the frame store
(`data/01_raw/sequences`, tracked by `data/01_raw/sequences.dvc`) and the scored
report (`data/08_reporting`, a cached `dvc.yaml` output). The viewer then renders
the **triage view**: the **To Review / Unlabel** split, a clickable
**per-organization** breakdown, and per-sequence tubes/boxes/crops on the right.
(DVC remote: `s3://pyro-vision-rd/dvc/temporal-model/triage/`.)

Everything below is only for **producing** new data — pulling fresh sequences
and re-scoring them.

---

## Producing: pull (annotator credentials required)

```bash
make install
cp .envrc.example .envrc     # fill in read-only annotator credentials (direnv loads it)
```

`.envrc` sets `ANNOTATOR_API_URL=https://annotationapi.pyronear.org` plus the
read-only login/password.

**Always test on a small subset first** (cheap, read-only):

```bash
make pull ARGS="--limit 3"     # smoke-test the client + store end-to-end
make pull ARGS="--limit 50"    # sanity-check the split on a real sample
```

Then the full backlog (~21k sequences, ~30 GB, a few hours). Use concurrency —
`--seq-workers N` pulls N sequences in parallel, `--workers M` downloads M frames
per sequence; keep `N × M` near the HTTP pool (~32), which the API tolerates:

```bash
make pull ARGS="--seq-workers 8 --workers 4"   # ~3 seq/s, polite (≈32 in-flight)
```

`make pull` runs the fetch, then `dvc add data/01_raw/sequences` + `dvc push`.
The pull is **incremental** (sequences already on disk are skipped) and writes
each `meta.json` last, so an interrupted run is safely re-pulled. Override the
stage with `--stage <processing_stage>` (default `ready_to_annotate`).

## Producing: score + push

```bash
make fetch-model                  # repo root: ensure ../api/models/model.zip exists
AWS_PROFILE=pyronear uv run dvc repro   # score the store → report + worklists
AWS_PROFILE=pyronear uv run dvc push    # share store + report so teammates can dvc pull
```

Override the split ad-hoc with `uv run temporal-triage score --threshold 0.5`,
or change `triage.threshold` in `params.yaml` (DVC re-runs `score`; re-bucketing
is free — the `probability` values don't change).

## CLI reference

```
temporal-triage pull   [--store DIR] [--stage STAGE] [--limit N]
                       [--page-size N] [--workers M] [--seq-workers N]
temporal-triage score  [--store DIR] [--output-dir DIR] [--model-zip ZIP]
                       [--threshold T] [--device cuda|mps|cpu]
```

## Outputs (`data/08_reporting/pyro-annotator/vit_dinov2_finetune/`)

The eval-viewer contract (so the viewer / eval tooling read it unchanged), plus
triage worklists. All rows join on `key` = `pyro-annotator_<sequence_id>`.

| File | What it holds |
|------|----------------|
| `results.json` / `results.parquet` | one row per sequence: `triage_score`, `triage_bucket`, `decision`, `probability`, `num_tubes_kept`, `trigger_frame_index`, org/camera/time. Parquet is the columnar twin for analysis. |
| `details/<key>.json` | full model output: tubes, per-frame bboxes, logits, calibrated probability, trigger, stabilized window. |
| `sequences/<key>.json` | frame paths (relative to `triage/`) + metadata — the viewer join. |
| `unlabeled.json` | low bucket: `sequence_ids` + a ready-to-send `bulk` payload (`false_positive_type: "unlabeled"`) — **written to disk only, never sent**. |
| `review.json` | high bucket: `sequence_ids` + scores, highest first. |
| `model_config.json` | model provenance + the `threshold` used. |
| `dropped.json` | skipped sequences + reason (`no_images`, `predict_failed`). |

## How it works (data layout)

```
data/01_raw/sequences/<org>/<camera>/seq_<id>/   # DVC-tracked frame store
    meta.json                                    #   identity + ordered frames (detection_id, bucket_key)
    images/detection_<id>.jpg
data/08_reporting/pyro-annotator/vit_dinov2_finetune/   # cached DVC output (the report above)
```

- **`pull`** (`pull.py` + `annotator_api.py`): login → page
  `GET /sequences/?processing_stage=ready_to_annotate` → per sequence fetch
  detections + signed image URLs → download into the store. Read-only.
- **`score`** (`score.py`): load each stored sequence into `core` `Frame`s, run
  `BboxTubeTemporalModel.predict()`, sequence score = max kept-tube probability,
  bucket at `threshold`.
- **`report.py`** writes the contract + worklists.

At full scale the loose store is ~300k DVC objects; the planned evolution is
tar-sharded frame storage — see the
[sharding design](../docs/specs/2026-06-16-triage-sharding-design.md).

## Tests

```bash
make test    # offline: mocked HTTP, fake store, stub model — no network, no Docker
```
