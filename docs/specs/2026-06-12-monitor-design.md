# monitor/: Production Decision Replay & Viewer

**Date:** 2026-06-12
**Status:** Approved

## Motivation

The temporal model API is deployed and alert-api calls it on every sequence
needing validation. Production stores the lean verdict only — a probability
plus the `version` provenance block — so nobody can see *why* the model kept
or discarded a sequence: no tubes, no per-frame boxes, no trigger frame.

`monitor/` answers "what did the model decide in production, and why?": it
imports recently scored sequences from alert-api, replays them through the
**exact** api + model release that scored them, captures the verbose tube
details, and serves them to the existing `viewer/`.

## Decisions (agreed in brainstorming)

1. **Replay through the pinned release Docker image, locally.** Each
   sequence records `temporal_api_version` (alert-api now returns the real
   tag, not `latest`). Release images on DockerHub
   (`pyronear/temporal-model-api:<tag>`) bake both the serving code and the
   exact `model.zip` (`api/Dockerfile` `COPY api/models/model.zip`), so one
   `docker pull` reproduces code + weights. No load or auth against the
   production instance; old versions stay replayable forever.
   (Alternatives rejected: calling the deployed API directly — adds prod
   load, needs the prod token, breaks on upgrade; running `core` directly —
   matches the model but not the api code path.)
2. **Reuse the Next.js eval viewer, no new UI.** Monitor writes the exact
   reporting contract `viewer/` already reads
   (`results.json`, `details/<key>.json`, `sequences/<key>.json`,
   `model_config.json` under
   `data/08_reporting/<source>/vit_dinov2_finetune/`) and the viewer points
   at it with `DATA_ROOT=../monitor`. Tube timeline, bbox overlay and
   stabilized crops come for free. Monitor-specific columns are additive.
3. **Frames are served to the pinned image via MinIO, uniformly.**
   Production sequences were scored by releases ≤ v0.3.1, which predate the
   local-frames feature (PR #49) — and an old image *silently ignores* an
   unknown `source: "local"` field and would hit S3 with those paths. A
   monitor-owned compose stack (pinned api image + MinIO) with frames
   uploaded under their original `bucket_key`s works identically against
   every version: one code path. A mount-the-store fast path for ≥ #49
   releases is a possible follow-up, not v1.
4. **Always request `?verbose=true&compute_trigger=true`.** Releases with
   PR #51 return `trigger_frame_index` / `trigger_tube_id` /
   `first_crossing_frame`; older pinned images ignore the unknown query
   param and the viewer renders "no trigger". Replay is offline, so the
   trigger search's latency cost is irrelevant.
5. **`stabilized_window` is recomputed client-side.** The verbose response
   does not include it. It is deterministic geometry (union of a tube's
   observed bboxes), so monitor derives it when writing
   `details/<key>.json`, keeping the viewer's crop panel working for every
   api version.
6. **Provenance and recorded score come from alert-api's `SequenceRead`**,
   which now exposes `temporal_model_score`, `temporal_model_version` and
   `temporal_api_version`. The recorded score doubles as a consistency
   check against the replayed probability.
7. **Credentials via `.envrc` (direnv), like `api/`.** Monitor reads
   `ALERT_API_URL`, `ALERT_API_LOGIN`, `ALERT_API_PASSWORD` from the
   environment; a committed `.envrc.example` documents them and the real
   `.envrc` stays untracked.
8. **DVC tracks both the store and the replay artifacts**, following the
   `train/`/`eval/` setup exactly: a per-package DVC repo (`.dvc/config`
   with `analytics = false` and an `s3remote` at
   `s3://pyro-vision-rd/dvc/temporal-model/monitor/`) and Kedro-style
   layer dirs. The sequence store is `dvc add`-tracked as
   `data/01_raw/sequences.dvc` (imports extend it, then `dvc add` +
   `dvc push`, like eval's `data/01_raw/pyro-annotator.dvc`); the replay
   inference is a `dvc.yaml` stage whose outputs are **cached and
   pushed** — a deliberate divergence from eval's `cache: false`
   reporting outs, because sharing is the point: a teammate runs
   `dvc pull` and opens the viewer with no alert-api credentials and no
   Docker. Import itself stays a CLI command outside the pipeline (it
   talks to a live API and is append-only, not reproducible).
9. **No `core` dependency.** Monitor talks HTTP and JSON and does small
   geometry; pulling torch/ultralytics for that would be waste.

## Architecture

A sixth uv package, `temporal_model.monitor`, exposing a `temporal-monitor`
CLI with two commands forming a pipeline, plus the existing viewer:

```
alert-api ──import──▶ data/01_raw/sequences/  ──dvc repro──▶ data/08_reporting/
                      (frames + meta.json,      (replay      (eval-viewer contract,
                       dvc add + push)           stage)       cached outs, dvc push)
                                                                    │
                                              viewer/ (DATA_ROOT=../monitor)
```

```
monitor/
├── pyproject.toml          # uv package: temporal-model-monitor
├── Makefile                # install | lint | format | test (mirrors siblings)
├── README.md
├── .envrc.example          # ALERT_API_URL / ALERT_API_LOGIN / ALERT_API_PASSWORD
├── .dvc/config             # s3remote, like train/ and eval/
├── .dvcignore
├── dvc.yaml                # replay stage (see below)
├── docker-compose.yml      # pinned api image + MinIO (no build:)
├── src/temporal_model/monitor/
│   ├── cli.py              # temporal-monitor import|replay
│   ├── alert_api.py        # login, sequences, detections client
│   ├── store.py            # sequence store read/write (meta.json)
│   ├── reconstruct.py      # frame selection + ROI, mirroring pyro-api
│   ├── stack.py            # compose lifecycle, MinIO upload, health checks
│   ├── replay.py           # /predict calls + consistency checks
│   ├── geometry.py         # stabilized_window recompute
│   └── report.py           # eval-viewer contract writers
├── tests/
└── data/                   # DVC-managed (gitignored content)
    ├── 01_raw/sequences/<org>/<camera>/seq_<id>/{meta.json, images/}
    ├── 01_raw/sequences.dvc
    └── 08_reporting/alert-api/vit_dinov2_finetune/...
```

The replay stage in `dvc.yaml`, in the house style (explicit code deps,
`uv run python -m` cmd):

```yaml
stages:
  replay:
    cmd: >-
      uv run python -m temporal_model.monitor.cli replay
      --store data/01_raw/sequences
      --output-dir data/08_reporting
      --trigger-image temporal-model-api:dev
    deps:
      - src/temporal_model/monitor/cli.py
      - src/temporal_model/monitor/store.py
      - src/temporal_model/monitor/reconstruct.py
      - src/temporal_model/monitor/stack.py
      - src/temporal_model/monitor/replay.py
      - src/temporal_model/monitor/geometry.py
      - src/temporal_model/monitor/report.py
      - docker-compose.yml
      - data/01_raw/sequences
    outs:
      - data/08_reporting        # cached → dvc push shares it
```

## `temporal-monitor import`

Modeled on vision-rd's `temporal-model-explorer` import, modernized:

- Login: `POST /api/v1/login/creds` → bearer token.
- Sequences: `GET /api/v1/sequences/all/fromdate?from_date=...&limit=100&offset=...`
  for each day in `--date-from/--date-to` (default: yesterday..today).
- Per sequence: `GET /api/v1/sequences/{id}/detections?limit=100&desc=false (paginated until a short page)`,
  download each detection's full frame via its presigned `url`, write
  `images/detection_<id>.jpg`.
- **Incremental:** a sequence already present in the store is skipped
  (`--force` re-downloads). The explorer re-downloaded everything; monitor
  runs recurringly.
- After an import, the store is synced with
  `dvc add data/01_raw/sequences && dvc push` (a `make import` convenience
  target wraps import + add + push).

`meta.json` extends the explorer's schema with everything replay needs:

```json
{
  "key": "alert-api_42307",
  "sequence_id": 42307,
  "source": "alert-api",
  "label": "smoke" | "fp" | "unknown",
  "label_detail": "wildfire_smoke" | "other_smoke" | "other" | null,
  "camera_id": 122, "camera_name": "...", "organization_id": 11,
  "organization_name": "...", "started_at": "...",
  "temporal_model_score": 0.9867,
  "temporal_model_version": "0.1.0",
  "temporal_api_version": "0.3.1",
  "frames": [
    {
      "file": "images/detection_14549.jpg",
      "detection_id": 14549,
      "created_at": "...",
      "bucket_key": "<original S3 key>",
      "bboxes": "<detection bboxes payload, verbatim>"
    }
  ]
}
```

Label mapping is fixed (no config): `is_wildfire` ∈
{`wildfire_smoke`, `other_smoke`} → `smoke`; `other` → `fp`; `null` →
`unknown`.

## `temporal-monitor replay`

Runs as the `replay` stage via `dvc repro` (directly invocable too); a
store change (new import) makes the stage stale, `dvc repro` re-runs it,
`dvc push` shares the artifacts.

1. **Group** stored sequences by `temporal_api_version`. Sequences with a
   null version (never scored, or pre-provenance) go to `dropped.json` with
   a reason; so do groups whose image tag cannot be pulled.
2. **Per group:** `docker pull pyronear/temporal-model-api:<version>`,
   `docker compose up` (api + MinIO), wait for `GET /health` to report
   `model_loaded: true`.
3. **Sanity check:** `/health.model_version` must equal each sequence's
   recorded `temporal_model_version`; a mismatch drops the sequence with
   reason `model_version_mismatch` (the image's baked model is not the one
   that scored it — should not happen, must not pass silently).
4. **Upload** the sequence's frames to MinIO under their original
   `bucket_key`s.
5. **Reconstruct the production call** exactly as alert-api's validation
   worker does (`pyro-api/src/app/services/validation.py`): distinct
   `bucket_key`s ordered by `created_at` ascending, truncated to the **last
   10**; skip (→ `dropped.json`) if fewer than 4 distinct frames;
   `roi_xyxyn` = union envelope of the kept detections' primary bboxes,
   `null` if none parse. The reconstruction logic mirrors pyro-api and is
   fixture-tested against it.
6. **Call** `POST /predict?verbose=true&compute_trigger=true` with
   `{bucket, frames, roi_xyxyn}`.
7. **Consistency check:** replayed `probability` vs recorded
   `temporal_model_score` (tolerance 1e-5: cross-hardware float noise). Recorded per sequence as
   `replay_matches`; mismatches are summarized at the end of the run.
   Known, accepted limitation: the detection set may have grown after the
   last production scoring, so reconstruction can legitimately differ —
   `replay_matches` makes that visible instead of silent.
8. **Write the reporting tree** — a single source named `alert-api`
   containing all organizations' rows (superseded per-org design after live
   use; see "Source naming in the viewer" below):
   - `results.json` — eval columns (`key`, `source`, `label`, `decision`,
     `outcome`, `score`, `probability`, `num_tubes_kept`,
     `trigger_frame_index`, `organization_name`, `camera_name`,
     `started_at`) carrying production's verdict (`decision`,
     `probability` = recorded score) plus monitor extras:
     `replayed_probability`, `replayed_decision`, `replay_matches`,
     `matched_window_frames`, `temporal_model_version`,
     `temporal_api_version`.
     `source` is always the fixed slug `"alert-api"`; `organization_name`
     carries the raw org name and is shown as a table column in the viewer.
   - `details/<key>.json` — verbose `details` reshaped to the eval
     `BboxTubeDetails` shape (`tubes.kept`, `decision`, `preprocessing`),
     with `stabilized_window` recomputed and trigger fields when the
     pinned version returns them.
   - `sequences/<key>.json` — `SequenceView` with frame paths relative to
     `monitor/` (e.g.
     `data/01_raw/sequences/<org>/<camera>/seq_<id>/images/...`).
     The store layout (per-org dirs) is unchanged; only the reporting
     tree root changes.
   - `model_config.json` — versions from `/health` + the decision block
     from a verbose response (the full training config is not available
     over HTTP; the viewer tolerates missing keys).

Outcome mapping reuses eval's semantics: `smoke`+keep → kept-smoke,
`fp`+keep → kept-fp, etc.; `unknown` → n/a.

## Source naming in the viewer

*This section supersedes the per-org design originally sketched here.*

All replayed sequences land in a single reporting tree at
`data/08_reporting/alert-api/vit_dinov2_finetune/`. The viewer's source
selector shows one entry ("alert-api") for monitor data instead of one
entry per organization. The `organization_name` field on each row carries
the raw org name and is rendered as a dedicated "organization" column in
the sequence table (shown only when the viewer is in monitor mode, detected
by the presence of `replayed_probability` on the rows). This design was
adopted after live use revealed that per-org sources added friction when
comparing sequences across organizations.

Frame paths inside `sequences/<key>.json` still point into the per-org
store dirs (`data/01_raw/sequences/<org>/<camera>/...`); the store layout
is unchanged.

## Viewing

```bash
cd viewer && DATA_ROOT=../monitor npm run dev
```

No structural viewer changes beyond the organization column and slim filter
bar (outcome chips and GT dropdown hidden in monitor mode). Eval flows are
untouched.

## Error handling

- Every skipped sequence lands in `dropped.json` with a machine-readable
  reason: `no_temporal_version`, `no_recorded_score`, `invalid_api_version`,
  `image_pull_failed`, `stack_unhealthy`, `model_version_mismatch`,
  `too_few_frames`, `no_images`, `predict_failed`.
- Import is resumable: a partially-downloaded sequence (images missing vs
  meta) is re-fetched on the next run.
- Replay failures on one sequence (HTTP error, timeout) are logged, the
  sequence is dropped with reason `predict_failed`, and the run continues.

## Conventions

- `monitor/` mirrors sibling packages: own `pyproject.toml` (uv, Python
  3.11+), `Makefile` with `install|lint|format|test`, `tests/`, README.
- Added to the root `Makefile` `PACKAGES` list and to the CI matrix in
  `.github/workflows/ci.yml`.
- Dependencies: `requests`, `pydantic` (schemas), `dvc[s3]>=3.56` (same
  pin as train/eval), `docker` via subprocess (no docker SDK). No torch,
  no `core`.

## Testing

All tests run offline (mocked HTTP, no Docker):

- `alert_api`: auth flow, pagination, endpoint URLs (mocked `requests`).
- `store`: meta.json round-trip, incremental skip, partial-download repair.
- `reconstruct`: frame selection (distinct, ordered, last-10, min-4) and
  ROI envelope against fixtures mirroring pyro-api's validation worker
  tests.
- `geometry`: stabilized_window union matches eval's values on a golden
  details fixture.
- `report`: written artifacts validate against golden files copied from an
  eval reporting tree (schema compatibility is the contract).
- `replay`: grouping, drop reasons, consistency check (stubbed predict).

The compose stack itself is exercised by a documented manual e2e recipe in
the README (import one day, replay, open viewer), not CI.

## Out of scope

- Local-frames fast path (`source: "local"`) for ≥ PR #49 releases.
- Any api/ or core/ changes (e.g. adding `stabilized_window` to verbose).
- Aggregate monitoring dashboards (alerting, drift charts over time) — the
  viewer's per-source performance cards are the v1 summary.
- pyro-annotator imports (eval already covers labeled offline data).
