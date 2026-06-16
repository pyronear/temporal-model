# triage/: Tar-Sharded Frame Storage

**Date:** 2026-06-16
**Status:** Implemented (`shards.py`, `pack`/`unpack` CLI)

## Implementation notes (deltas from the original draft)

- **The report is sharded too.** At full scale `details/` + `sequences/` are
  ~43k tiny files — not "small". `pack` bundles per-sequence `details.json` +
  `sequence.json` into `report/` shards; only the ~6 aggregate files
  (`results.json`/`.parquet`, worklists, `model_config`, `dropped`) stay loose.
- **Frames and predictions are separate shard sets** (`frames/` append-only and
  model-independent; `report/` rebuilt per run) — a re-score never re-packs the
  26 GB of frames.
- **Every prediction is tagged with `model_version`** (the release version from
  the model.zip manifest, e.g. `0.2.0`), so a future re-score's predictions stay
  distinguishable. Multi-version report namespacing is deferred until a second
  model run actually happens.
- **No `dvc.yaml` pipeline.** The 247k-file store is impractical as a stage dep
  and the workflow is staged/manual, so the single DVC-tracked artifact is
  `data/02_shards`, added with `dvc add`.

## Motivation

`triage` stores pulled frames as loose files under
`data/01_raw/sequences/<org>/<camera>/seq_<id>/images/*.jpg` and DVC-tracks the
directory. At the 500-sequence scale this is ~6,934 S3 objects — fine. At the
full `ready_to_annotate` backlog (~21,489 sequences, ~301k frames) it becomes
**~300k loose S3 objects / ~30 GB**: DVC content-addresses one object per file,
so push/pull is dominated by per-object overhead, the single `.dir` index grows
to ~300k entries (slow `status`/`checkout`), and per-request S3 cost climbs.

The fix is to **pack frames into tar shards** so DVC tracks tens of objects
instead of hundreds of thousands. The scored outputs (`results.parquet` +
small JSON) are tiny and stay loose — only frames are sharded.

## Decisions

1. **Append-only sealed shards.** Each `pack` run writes the sequences not yet
   in any shard into one or more **new, immutable** tar files
   (`shard_0001.tar`, `shard_0002.tar`, …); existing shards are never rewritten.
   A broad incremental pull therefore re-pushes only its new shard(s), not the
   whole set. (Rejected: `sequence_id % N` bucketing — even and deterministic,
   but adding any sequence rewrites a ~1 GB bucket tar, so a wide incremental
   pull re-pushes ~everything. Rejected: keep loose — the ~300k-object problem.)
2. **Only frames are sharded.** Each shard holds, per sequence, both the frames
   and the `meta.json` (so a shard is self-contained). The scored report
   (`results.parquet`, `results.json`, `details/`, `sequences/`,
   `model_config.json`) stays loose under `data/08_reporting/` — it is small and
   the viewer reads per-key files.
3. **The loose store is a local, regenerable cache.** `data/01_raw/sequences/`
   is gitignored and **not** DVC-tracked. It is produced either by `pull`
   (producer) or by `unpack` (consumer). The DVC source of truth for frames is
   `data/02_shards/`.
4. **A manifest drives incrementality.** `data/02_shards/manifest.json` maps
   `sequence_id → shard_name` (plus `next_index`). `pack` packs only sequences
   absent from the manifest; `unpack`/tooling use it to know what exists.
5. **Migration is a one-time, deliberate cleanup.** Cutting over from the loose
   layout orphans the already-pushed loose frame objects. We wipe the
   `triage/` remote prefix and re-push under the shard layout rather than rely
   on `dvc gc` (see Migration).

## Layout

```
data/
├── 02_shards/                      # DVC-tracked (cache:true) → pushed. Source of truth for frames.
│   ├── shard_0001.tar              # immutable; each holds seq_<id>/{meta.json, images/*.jpg}
│   ├── shard_0002.tar
│   └── manifest.json               # {next_index, sequences: {<id>: "shard_0001.tar", ...}}
├── 01_raw/sequences/               # LOCAL ONLY (gitignored, not DVC-tracked).
│   └── <org>/<camera>/seq_<id>/…   #   produced by pull OR unpack; consumed by score + viewer.
└── 08_reporting/…                  # scored outputs (parquet + JSON), loose, unchanged.
```

Shard tar member paths mirror the store's per-sequence layout so `unpack`
restores it byte-for-byte:
`<org_slug>/<camera_slug>/seq_<id>/meta.json` and `.../images/detection_*.jpg`.
Tars are **uncompressed** (JPEGs are already compressed). Each shard is sealed
at a target size (`SHARD_TARGET_BYTES`, default ~1 GB ≈ ~700 sequences); a
`pack` run that exceeds it rolls to the next index.

## Commands (two new)

- **`temporal-triage pack`** — read `data/01_raw/sequences`, select sequences
  absent from `manifest.json`, append them into new sealed shard(s), update the
  manifest. Idempotent: re-running with nothing new is a no-op. Does not delete
  the loose copies (they stay as the local working set).
- **`temporal-triage unpack`** — extract every shard in `data/02_shards` into
  `data/01_raw/sequences` (skips sequences already materialized). Reconstitutes
  the loose store for `score` and the viewer from `dvc pull`-ed shards alone.

`score`, `pull`, `report`, and the viewer are **unchanged** — they keep reading
the loose store by path.

## Workflows

**Producer (annotator creds):**
```
make pull ARGS="--limit N"     # loose store (incremental, parallel downloads)
temporal-triage pack           # new sequences → new sealed shard(s) + manifest
dvc add data/02_shards         # track shards
dvc push                       # upload only the new shard(s)
```

**Consumer (no creds, no Docker):**
```
dvc pull                       # fetch shards (tens of objects)
temporal-triage unpack         # shards → loose store
dvc repro                      # score → report   (or: temporal-triage score)
cd ../viewer && DATA_ROOT=../triage npm run dev
```

## DVC pipeline changes

- Untrack the loose store: remove `data/01_raw/sequences.dvc`.
- Track shards: `dvc add data/02_shards` (cache:true → pushed).
- `score` stage keeps `data/01_raw/sequences` as a **path dependency** (DVC
  hashes it); the loose store is materialized by `pull` or `unpack` before
  `score` runs. (Optionally an `unpack` stage with a `cache:false` output, so a
  pure `dvc repro` on a fresh checkout unpacks then scores — to be decided in
  implementation.)

## Object-count impact (full backlog, ~21,489 seq / ~301k frames)

| Layout | Frame S3 objects | Incremental pull of +500 seq | `.dir` size |
|--------|------------------|------------------------------|-------------|
| Loose (today) | ~300k | cheap (only new frames push) | ~300k entries |
| Append-only shards | **~30–40 tars** | **1 new ~1 GB tar** | tiny |

## Migration (one-time)

1. `temporal-triage pack` the current 500-sequence loose store → `shard_0001.tar` + manifest.
2. `aws s3 rm --recursive s3://pyro-vision-rd/dvc/temporal-model/triage/` — wipe the prefix (orphaned loose frames + report; both are regenerable).
3. Remove `data/01_raw/sequences.dvc`; `dvc add data/02_shards`; `dvc commit score` (report); `dvc push`.
4. Amend/replace the loose-layout pointer commit (`4ee716d`) so **no retained
   commit references the orphaned loose frame objects** — this avoids `dvc gc`
   reachability footguns entirely on the shared remote.

## Out of scope

- Compression of tars (JPEGs are already compressed).
- Deleting loose frames after `pack` (kept as the local working set; gitignored).
- Streaming frames directly from tars into the model (would require changing
  `core`'s file-path `Frame` contract; `unpack` to loose files is non-invasive).
- Not storing frames at all (reference-only + on-demand refetch) — leaner, but
  couples reuse to live annotator access; revisit separately if shard storage
  proves heavy.
