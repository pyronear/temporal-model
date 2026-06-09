# Smooth `model.zip` provisioning for `make serve`

**Status:** Designed (2026-06-09)
**Author:** Chouffe

## Problem

A fresh `git clone` cannot run `make serve`. The flow is:

```
make serve → cd api && docker compose up --build → Dockerfile: COPY api/models/model.zip /models/model.zip
```

`api/models/` is gitignored, so on a clean clone the file is absent and the
Docker build fails with a cryptic checksum error:

```
ERROR [api stage-0 9/9] COPY api/models/model.zip /models/model.zip
failed to compute cache key ... "/api/models/model.zip": not found
```

There is no obvious signal to a newcomer that they must first obtain a
`model.zip`, nor where to get one.

## Constraints & context

- **Audience: anyone (public).** The smooth path must need no credentials.
- A public HuggingFace release path **already exists**:
  `temporal_model.api.release fetch --version <v> --output <path>` downloads
  `model.zip` from `pyronear/temporal-model` at revision `v<version>` with no
  token. `scripts/release-api.sh` already uses it (fetch + bake into image).
- DVC also exists (`train/`, `eval/`) but points at a **private** S3 bucket
  (`s3://pyro-vision-rd/...`) requiring AWS creds — disqualified for the public
  goal.
- Current released version: `0.1.0` (HF tag `v0.1.0`, git tag `v0.1.0`).
- `docker-compose.yml` volume-mounts `./models:/models`, which shadows the
  baked `COPY` at runtime. So for local `make serve` the `COPY` is effectively a
  build-time landmine; the runtime model comes from the mount.

## Goal

A fresh clone either serves cleanly or **fails fast with a clear, actionable
message** — never the raw docker `COPY ... not found` error.

## Design

Reuse the existing public HF `fetch`. No new download code — wiring + a guard.

### 1. Root `Makefile` — add `fetch-model`

```make
MODEL_VERSION ?= 0.1.0

fetch-model: ## download the released model.zip from HuggingFace (no creds)
	cd api && uv run python -m temporal_model.api.release \
	    fetch --version $(MODEL_VERSION) --output models/model.zip
```

This is exactly the fetch half of `scripts/release-api.sh`, reused. Pinned and
reproducible; override with `make fetch-model MODEL_VERSION=x.y.z`.

### 2. Root `Makefile` — guard `serve`

Before invoking docker compose, `serve` checks for `api/models/model.zip`. If
missing, it prints an actionable message and exits non-zero:

```
model.zip not found — run 'make fetch-model' (downloads v<MODEL_VERSION> from HuggingFace, no credentials)
```

This replaces the cryptic docker failure with a one-line instruction and stops
before the build starts.

### 3. Dockerfile — no change

The `COPY api/models/model.zip /models/model.zip` remains correct for the
released-image path (`release-api.sh` bakes the model in). With the `serve`
guard in place the `COPY` no longer fires against an empty clone. The existing
mount/COPY redundancy is intentionally left untouched (out of scope).

### 4. Docs

`api/README.md` and root `README.md`: replace the "place a `model.zip` under
`api/models/`" instructions with "run `make fetch-model` before `make serve`
(downloads v0.1.0 from HuggingFace, no credentials needed)."

## Out of scope (explicitly rejected)

- **DVC** for the dev path — private bucket, needs creds; fails the public goal.
- **Auto-fetch / in-Docker download** — an explicit `make fetch-model` target
  was chosen over `make serve` silently downloading or the Dockerfile fetching
  at build time.

## Verification

1. `rm -f api/models/model.zip && make serve` → fast, clear failure naming
   `make fetch-model` (not a docker error).
2. `make fetch-model` → pulls v0.1.0 from HF to `api/models/model.zip`, no creds.
3. `make serve` (after fetch) → builds and runs.
