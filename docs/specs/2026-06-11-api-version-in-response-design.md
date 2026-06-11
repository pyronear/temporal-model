# API: Code Version in `/predict` Responses

**Date:** 2026-06-11
**Status:** Approved

## Motivation

A `/predict` result is produced by two independently-versioned artifacts: the
packaged model (`model.zip`, pinned by `api/MODEL_VERSION`) and the serving
code (the Docker image, tagged from the repo git tag — the two were
deliberately decoupled in PR #44). Today the response only names the model
(`model: {name, version}`); the code version is invisible at runtime, so a
stored result cannot be traced back to the exact image that produced it.

Worse, the code version has no reliable runtime source at all:
`FastAPI(version="0.1.0")` in `app.py` and `pyproject.toml`'s `0.1.0` are both
already stale against the `v0.2.0` repo tag.

Goal: every `/predict` response self-describes **model + code** so alert-api
can persist one provenance object per detection.

## Decisions (agreed in brainstorming)

1. **One grouped `version` block** in the response, holding both identities.
   This is a **breaking change** to the `/predict` contract (the `model`
   block moves) — accepted; alert-api integration can absorb it.
2. **Semver only** for the code identity — the release version that matches
   the Docker image tag. No git SHA (tags are immutable; the SHA is
   recoverable from the tag).
3. **Drop `model.name`** from `/predict`. `name` is `manifest.variant` and
   the repo is ViT-only — a single variant. The model's release identity is
   its version alone (`v<version>` revision on HF `pyronear/temporal-model`,
   whose manifest carries variant, backbone, and full provenance). The block
   becomes two flat `string | null` fields. `/health` keeps `model_name` —
   it is an ops endpoint, not a stored contract.
4. **Version source: Docker build arg** (not pyproject, not a pin file). CI
   already computes `VERSION` from the git tag; it is baked into the image
   as an env var, so the reported version is guaranteed to equal the image
   tag with zero bump discipline. Local runs report `null` — honest for a
   working tree, with the same "not a stamped release artifact" meaning as
   the existing `null` for legacy model packages. Overridable via the env
   var when a local value is wanted.

## API contract

### `/predict` response (breaking)

```json
{
  "is_smoke": true,
  "probability": 0.93,
  "version": {
    "api": "0.2.0",
    "model": "0.2.0"
  },
  "details": { "...": "unchanged, verbose only" }
}
```

- `version.api`: `string | null`. The serving-code release version, identical
  to the Docker image tag `pyronear/temporal-model-api:<version>`. `null`
  when not running a release image (local uvicorn, tests).
- `version.model`: `string | null`. The packaged model's `model_version`
  (formerly `model.version`); `null` for legacy unstamped packages.
- The top-level `model` block is **removed**. `model.name` has no
  replacement in `/predict` (decision 3).
- `details` is unaffected.

### `/health` (additive)

Gains `api_version: string | null` from the same setting. Existing flat
fields (`model_name`, `model_version`) stay as they are:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "vit",
  "model_version": "0.2.0",
  "api_version": "0.2.0"
}
```

### OpenAPI document

`FastAPI(version=...)` stops hardcoding `"0.1.0"` and uses
`settings.api_version or "dev"` (the OpenAPI `info.version` field requires a
string).

## Version source & plumbing

- **`settings.py`** — new field `api_version: str | None = None`, env
  `TEMPORAL_API_VERSION` (existing prefix convention).
- **`api/Dockerfile`** —

  ```dockerfile
  ARG VERSION
  ENV TEMPORAL_API_VERSION=${VERSION}
  ```

- **`.github/workflows/push.yml`** — the "Resolve versions" step already
  exports `VERSION` from the git tag; the build-push step gains

  ```yaml
  build-args: |
    VERSION=${{ env.VERSION }}
  ```

- **Local runs** — env var unset → `version.api: null`. Set
  `TEMPORAL_API_VERSION` (shell or `api/.envrc`) to override. No
  `git describe` automation in `make serve` (a dirty tree would
  misrepresent itself).

## Code changes (all in `api/`)

| File | Change |
|---|---|
| `schemas.py` | `ModelInfo` → `Version {api: str \| None, model: str \| None}`; `PredictResponse.model` → `PredictResponse.version`; `to_response()` signature: `name`/`version` params → `api_version`/`model_version` |
| `app.py` | Pass `settings.api_version` into `to_response`; add `api_version` to `HealthResponse` and `/health`; `FastAPI(version=settings.api_version or "dev")` |
| `settings.py` | `api_version: str \| None = None` |
| `Dockerfile` | `ARG VERSION` → `ENV TEMPORAL_API_VERSION` |
| `.github/workflows/push.yml` | `build-args: VERSION=…` |
| `docs/model-versioning.md` | §1 table: add the runtime-reported `version.api` row |
| `api/README.md` | Update response examples |

`ModelRunner` still reads `name` from the manifest for `/health`; nothing
changes in `model_runner.py`.

## Testing

**Schemas** (`test_schemas.py`):

- `to_response` nests both versions under `version` and carries each through
  independently, including `None` for either.
- No `model` key in the serialized response.

**App** (`test_app.py` / existing route tests):

- With `api_version` configured: `/predict` returns
  `version == {"api": <value>, "model": <runner version>}`; `/health`
  includes `api_version`.
- With it unset (default test settings): `version.api` is `null`.
- Existing assertions on `response["model"]` updated to the new path —
  expected churn from the accepted breaking change.

**Not testable in-repo:** the build-arg → env plumbing (exercised by the
release workflow; verified by hitting `/health` on the next released image).

## Out of scope

- Git SHA / build metadata in the response (decision 2).
- Restructuring `/health` (ops endpoint, additive only).
- A CI check that pyproject version matches the tag — pyproject's version is
  no longer load-bearing anywhere; syncing it is cosmetic and can be done
  opportunistically.
- Response header (`X-API-Version`) — the body block is the stored contract.
