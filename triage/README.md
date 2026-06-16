# 🗂️🔥 triage — annotation-backlog triage

Pulls the **unannotated** backlog from the pyro-annotator
(https://annotator.pyronear.org) **read-only**, scores every sequence with the
temporal smoke classifier in-process, and splits it at a threshold (default
**0.35**):

- **low (`< 0.35`)** → `unlabeled.json`: a read-only worklist (sequence ids + a
  ready-to-send `bulk` body) to later mark these as the `unlabeled`
  false-positive type. triage **never writes** to the annotator.
- **high (`>= 0.35`)** → the eval-viewer contract, reviewed locally in
  [`viewer/`](../viewer).

Design: [`docs/specs/2026-06-16-triage-design.md`](../docs/specs/2026-06-16-triage-design.md)

## Read-only guarantee

The only non-GET request triage sends to the annotator is the login POST that
mints a bearer token. The HTTP client has no patch/put/delete/non-login-post
method (enforced by `tests/test_annotator_api.py`). The `bulk` payload in
`unlabeled.json` is written to disk only — applying it is a separate human step.

## Pulling (annotator credentials required)

```bash
make install
cp .envrc.example .envrc     # fill in read-only annotator credentials
```

**Test on a small subset first** (cheap, read-only):

```bash
make pull ARGS="--limit 3"     # smoke-test the client + store
make pull ARGS="--limit 50"    # sanity-check the split on a real sample
make pull                      # full backlog
```

`make pull` runs the fetch, then `dvc add data/01_raw/sequences` + `dvc push`.

## Scoring

```bash
make fetch-model                  # repo root: ensure ../api/models/model.zip exists
uv run dvc repro                  # score the store, write the report + worklists
cd ../viewer && DATA_ROOT=../triage npm run dev   # review the high bucket
```

Override the split ad-hoc: `uv run temporal-triage score --threshold 0.5`, or
change `triage.threshold` in `params.yaml` (DVC re-runs `score`).

## Outputs (`data/08_reporting/pyro-annotator/vit_dinov2_finetune/`)

- `results.json`, `details/<key>.json`, `sequences/<key>.json`,
  `model_config.json` — the eval-viewer contract (all scored sequences;
  `triage_bucket` / `decision` distinguish review vs unlabel).
- `unlabeled.json` — low bucket: ids + ready-to-send `bulk` payload (never sent).
- `review.json` — high bucket: ids + scores, highest first.
- `dropped.json` — skip reasons (`no_images`, `predict_failed`).

## Tests

```bash
make test    # offline: mocked HTTP, fake store, stub model — no network, no Docker
```
