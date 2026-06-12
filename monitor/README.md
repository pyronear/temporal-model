# 🔎🔥 monitor — production decision replay

Answers "what did the temporal model decide in production, and why?".
Imports sequences scored by the deployed API (alert-api records the
probability + `version` provenance per sequence), replays them through the
**exact** pinned release image with `verbose=true&compute_trigger=true`, and
writes the eval-viewer contract so tubes/boxes/crops are explorable in
[`viewer/`](../viewer).

Design: [`docs/specs/2026-06-12-monitor-design.md`](../docs/specs/2026-06-12-monitor-design.md)

## Setup

```bash
make install                 # uv sync
cp .envrc.example .envrc     # fill in alert-api credentials (direnv loads it)
dvc pull                     # optional: fetch the shared store + reports
```

## Workflow

```bash
make import                                  # 1. fetch new sequences (incremental),
                                             #    dvc add + push the store
make import ARGS="--date-from 2026-06-01 --date-to 2026-06-10"  # backfill a range
uv run dvc repro                             # 2. replay through pinned releases
uv run dvc push                              # 3. share the artifacts
cd ../viewer && DATA_ROOT=../monitor npm run dev   # 4. browse at localhost:3000
```

`replay` groups sequences by their recorded `temporal_api_version`, runs
`pyronear/temporal-model-api:<tag>` (model.zip baked in) + a throwaway MinIO
on ports 18000/19000 (offset from `api/`'s 8000/9000), uploads each
sequence's frames under their original S3 keys, and reconstructs the exact
production call (last ≤10 distinct frames oldest-first, ROI = envelope of
the detections' primary bboxes — mirrors pyro-api's validation worker).
Docker must be running; each version group costs one image pull.

## Outputs (`data/08_reporting/<org>/vit_dinov2_finetune/`)

- `results.json` — eval columns + monitor extras: `recorded_probability`
  (what production stored), `replay_matches` (|Δ| ≤ 1e-6),
  `matched_window_frames` (when a mismatch is window drift: the sequence
  length at which production's recorded score is reproduced; null = no window
  matched, genuine drift), `temporal_model_version`, `temporal_api_version`.
- `details/<key>.json` — tubes in the eval shape; `stabilized_window` is
  recomputed client-side; trigger fields appear for releases shipping
  `compute_trigger` (older images ignore the flag → "no trigger" in the
  viewer).
- `sequences/<key>.json`, `model_config.json`, `dropped.json` (skip reasons:
  `no_temporal_version`, `image_pull_failed`, `stack_unhealthy`,
  `model_version_mismatch`, `too_few_frames`, `no_images`, `predict_failed`).

A `replay_matches: false` row means the reconstruction diverged from the
recorded score. When this happens, the replay automatically probes candidate
windows (ascending distinct-frame counts from MIN_FRAMES) to check whether
the mismatch is window drift — production scores a sequence early and stops
once it validates, while detections keep arriving. If a shorter prefix of the
sequence reproduces the recorded score, `matched_window_frames` is set to
that count (window drift, not model drift); if no prefix matches, it remains
null (genuine drift — see the spec).

## Tests

```bash
make test    # offline: mocked HTTP, fake docker stack — no Docker needed
```
