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

The scored set is pushed to the DVC S3 remote as **tar shards** (~36 objects,
not ~290k loose files). Browsing needs only S3 read access (`AWS_PROFILE=pyronear`)
— no annotator credentials, no `model.zip`, no Docker:

```bash
cd triage
make install                                   # uv sync (brings dvc[s3])
AWS_PROFILE=pyronear uv run dvc pull            # fetch data/02_shards from S3
uv run temporal-triage unpack                  # tars → loose store + report
( cd ../viewer && npm install )                # first time only
make viewer                                    # browse at http://localhost:3000
```

`dvc pull` fetches `data/02_shards` (frame + report tars + manifests + the small
aggregate files); `unpack` restores the loose `data/01_raw/sequences` (frames)
and `data/08_reporting/.../` (per-key report) so the viewer reads them by path.
The viewer then renders the **triage view**: the **To Review / Unlabel** split, a
clickable **per-organization** breakdown, the threshold slider + sweep table, and
per-sequence tubes/boxes/crops. (DVC remote:
`s3://pyro-vision-rd/dvc/temporal-model/triage/`.)

Everything below is only for **producing** new data.

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

The pull is **incremental** (sequences already on disk are skipped) and writes
each `meta.json` last, so an interrupted run is safely re-pulled. Override the
stage with `--stage <processing_stage>` (default `ready_to_annotate`). The loose
store is local-only (gitignored, not DVC-tracked) — it's shared via shards below.

## Producing: score → pack → push

```bash
make fetch-model                              # repo root: get ../api/models/model.zip
uv run temporal-triage score                  # score store → report + worklists (uses GPU if present)
uv run temporal-triage pack                   # tar-shard store + report → data/02_shards
AWS_PROFILE=pyronear uv run dvc add data/02_shards
AWS_PROFILE=pyronear uv run dvc push          # share ~36 objects so teammates can dvc pull
```

Override the split ad-hoc with `temporal-triage score --threshold 0.5`;
re-bucketing is free (the `probability` values don't change). `pack` is
incremental for frames (model-independent, append-only) and rebuilds the report
shards (predictions are per model run); see the
[sharding design](../docs/specs/2026-06-16-triage-sharding-design.md).

## CLI reference

```
temporal-triage pull    [--store DIR] [--stage STAGE] [--limit N]
                        [--page-size N] [--workers M] [--seq-workers N]
temporal-triage score   [--store DIR] [--output-dir DIR] [--model-zip ZIP]
                        [--threshold T] [--device cuda|mps|cpu]
temporal-triage pack    [--store DIR] [--report-dir DIR] [--shards DIR] [--target-bytes N]
temporal-triage unpack  [--shards DIR] [--store DIR] [--report-dir DIR]
```

## Outputs (`data/08_reporting/pyro-annotator/vit_dinov2_finetune/`)

The eval-viewer contract (so the viewer / eval tooling read it unchanged), plus
triage worklists. All rows join on `key` = `pyro-annotator_<sequence_id>`.

| File | What it holds |
|------|----------------|
| `results.json` / `results.parquet` | one row per sequence: `triage_score`, `triage_bucket`, `decision`, `model_version` (e.g. `0.2.0`), `probability`, `num_tubes_kept`, `trigger_frame_index`, org/camera/time. Parquet is the columnar twin for analysis. |
| `details/<key>.json` | full model output: tubes, per-frame bboxes, logits, calibrated probability, trigger, stabilized window. |
| `sequences/<key>.json` | frame paths (relative to `triage/`) + metadata — the viewer join. |
| `unlabeled.json` | low bucket: `sequence_ids` + a ready-to-send `bulk` payload (`false_positive_type: "unlabeled"`) — **written to disk only, never sent**. |
| `review.json` | high bucket: `sequence_ids` + scores, highest first. |
| `model_config.json` | model provenance + the `threshold` used. |
| `dropped.json` | skipped sequences + reason (`no_images`, `predict_failed`). |

## How it works (data layout)

```
data/01_raw/sequences/<org>/<camera>/seq_<id>/   # loose store — LOCAL only (gitignored)
    meta.json                                    #   identity + ordered frames (detection_id, bucket_key)
    images/detection_<id>.jpg
data/08_reporting/pyro-annotator/vit_dinov2_finetune/   # loose report — LOCAL only
data/02_shards/                                  # the DVC-tracked, pushed artifact
    frames/shard_*.tar + manifest.json           #   per-sequence images+meta (append-only, model-independent)
    report/shard_*.tar + manifest.json           #   per-sequence details+sequence-view (per model run)
    results.json, results.parquet, *.json        #   aggregate report files, loose
```

- **`pull`** (`pull.py` + `annotator_api.py`): login → page
  `GET /sequences/?processing_stage=ready_to_annotate` → per sequence fetch
  detections + signed image URLs → download into the store. Read-only.
- **`score`** (`score.py`): load each stored sequence into `core` `Frame`s, run
  `BboxTubeTemporalModel.predict()`, sequence score = max kept-tube probability,
  bucket at `threshold`; tag each prediction with the release `model_version`.
- **`report.py`** writes the eval-viewer contract + worklists.
- **`shards.py`** (`pack`/`unpack`): the loose store (~247k files) + report (~43k
  files) would be a DVC object explosion, so `pack` bundles per-sequence data into
  ~36 tar objects and `unpack` restores the loose tree. Frames and predictions
  have separate lifecycles (frames immutable across model runs, predictions
  per-run), so they get separate tars — re-scores never re-pack the 26 GB of
  frames. See the [sharding design](../docs/specs/2026-06-16-triage-sharding-design.md).

`triage` deliberately has **no `dvc.yaml` pipeline**: the store is too many files
to hash as a stage dep, and the workflow is inherently staged (pull needs
credentials, score needs a GPU). The single DVC-tracked artifact is
`data/02_shards`, added with `dvc add`.

## Tests

```bash
make test    # offline: mocked HTTP, fake store, stub model — no network, no Docker
```
