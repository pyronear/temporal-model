# 🔎🔥 monitor — production decision replay

Answers "what did the temporal model decide in production, and why?".
Imports sequences scored by the deployed API (alert-api records the
probability + `version` provenance per sequence), replays them through the
**exact** pinned release image with `verbose=true&compute_trigger=true`, and
writes the eval-viewer contract so tubes/boxes/crops are explorable in
[`viewer/`](../viewer).

Design: [`docs/specs/2026-06-12-monitor-design.md`](../docs/specs/2026-06-12-monitor-design.md)

## Just browsing? Pull the shared data (no credentials, no Docker)

The store and the replay artifacts are pushed to the DVC remote, so viewing
what production decided requires neither alert-api credentials nor Docker:

```bash
make install                                       # uv sync (brings dvc[s3])
uv run dvc pull                                    # fetch store + reports from S3
cd ../viewer && DATA_ROOT=../monitor npm run dev   # browse at localhost:3000
```

Everything below is only for IMPORTING new sequences and RE-RUNNING the
model — the producer workflow.

## Importing new data (alert-api credentials required)

```bash
make install
cp .envrc.example .envrc     # fill in alert-api credentials (direnv loads it)
```

```bash
make import                                  # 1. fetch new sequences (incremental),
                                             #    dvc add + push the store
make import ARGS="--date-from 2026-06-01 --date-to 2026-06-10"  # backfill a range
make import ARGS="--all-orgs --exclude-org pyroadmins"  # admin token: every org, skipping CI org
```

The alert-api listing (`/sequences/all/fromdate`) is limited to the authenticated account's own
organization; `--all-orgs` scans the global sequence-id space instead (admin token required), so
every organization (sdis-07, sdis-77, ...) lands in the single `alert-api`
source, distinguishable via the organization column/filter in the viewer.
`--exclude-org` skips an organization (accepts a slug or raw name, e.g. `--exclude-org pyroadmins`
for the CI camera org); pass it multiple times to exclude several. Exclusion affects future imports
only — already-imported directories must be removed manually.

## Replaying (Docker required)

```bash
uv run dvc repro                             # 2. replay through pinned releases
uv run dvc push                              # 3. share store + artifacts so
                                             #    teammates can just `dvc pull`
cd ../viewer && DATA_ROOT=../monitor npm run dev   # 4. browse at localhost:3000
```

`replay` groups sequences by their recorded `temporal_api_version`, runs
`pyronear/temporal-model-api:<tag>` (model.zip baked in) + a throwaway MinIO
on ports 18000/19000 (offset from `api/`'s 8000/9000), uploads each
sequence's frames under their original S3 keys, and reconstructs the exact
production call (last ≤10 distinct frames oldest-first, ROI = envelope of
the detections' primary bboxes — mirrors pyro-api's validation worker).
Docker must be running; each version group costs one image pull.

## Outputs (`data/08_reporting/alert-api/vit_dinov2_finetune/`)

All organizations land in one reporting tree; organizations are a table column in the viewer.

- `results.json` — eval columns carrying PRODUCTION's verdict (`decision`,
  `probability` = the recorded score) + monitor extras: `replayed_probability`
  / `replayed_decision` (the local re-run), `replay_matches` (|Δ| ≤ 1e-5),
  `matched_window_frames` (when a mismatch is window drift: the sequence
  length at which production's recorded score is reproduced; null = no window
  matched, genuine drift), `temporal_model_version`, `temporal_api_version`.
- `details/<key>.json` — tubes in the eval shape; `stabilized_window` is
  recomputed client-side; trigger fields appear for releases shipping
  `compute_trigger` (older images ignore the flag → "no trigger" in the
  viewer).
- `sequences/<key>.json`, `model_config.json`, `dropped.json` (skip reasons:
  `no_temporal_version`, `no_recorded_score`, `invalid_api_version`,
  `image_pull_failed`, `stack_unhealthy`, `model_version_mismatch`,
  `too_few_frames`, `no_images`, `predict_failed`).

A `replay_matches: false` row means the reconstruction diverged from the
recorded score. When this happens, the replay automatically probes candidate
windows (ascending distinct-frame counts from MIN_FRAMES) to check whether
the mismatch is window drift — production scores a sequence early and stops
once it validates, while detections keep arriving. If a shorter prefix of the
sequence reproduces the recorded score, `matched_window_frames` is set to
that count (window drift, not model drift); if no prefix matches, it remains
null (genuine drift — see the spec).

## Trigger frames

Sequences scored by api v0.3.1 and earlier predate the `compute_trigger`
feature, so their replays produce no `trigger_frame_index` or
`first_crossing_frame` data. The enrichment pass fills these fields using a
newer serving image (the same `model.zip` baked in) while keeping the pinned
replay authoritative for tubes and probability.

Pass `--trigger-image <image>` to `replay` (or leave the default in dvc.yaml).
The enrichment probability must reproduce the pinned replay's within
`SCORE_TOLERANCE` (1e-5) — a disagreement means the scoring path changed and
the trigger fields are left empty for that sequence.

Build the dev image from the repo root:

```bash
make fetch-model MODEL_VERSION=0.2.0   # repo root: same model production runs
docker build -f api/Dockerfile --build-arg VERSION=dev -t temporal-model-api:dev .
```

Teammates who only need the artifacts can skip the build and use `dvc pull`
instead of `dvc repro`.

## Tests

```bash
make test    # offline: mocked HTTP, fake docker stack — no Docker needed
```
