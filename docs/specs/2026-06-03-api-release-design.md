# API Release — Design

**Date:** 2026-06-03
**Status:** Draft
**Scope:** How a tagged release produces a self-contained API Docker image with a
specific versioned `model.zip` **baked in**, sourced from **HuggingFace**. Covers
the locally-runnable `release` CLI (`fetch`/`publish`), the Dockerfile changes, the
adapted `push.yml` workflow, and the one-time bootstrap of the first artifact.
Builds on [`2026-06-03-model-versioning-design.md`](2026-06-03-model-versioning-design.md).
The **packaging stage** that *produces* the `model.zip` from a trained checkpoint
is the `train` DVC `package` stage (implemented in
[`2026-06-03-packaging-pipeline-design.md`](2026-06-03-packaging-pipeline-design.md)),
out of scope here — this spec's precondition is "a `model.zip` has been
**published** (version-stamped at publish time) to the HuggingFace repo at
revision `v<version>`."

## Goal

Turn a `v<version>` git tag into an immutable, self-contained image
`pyronear/temporal-model-api:<version>` that already contains the matching
`model.zip` — no runtime model mount, no model fetch at runtime. The model-fetching
and version-assertion logic lives in a small CLI that runs **locally with your HF
token** exactly as it runs in CI.

## Where the model lives — HuggingFace

The packaged `model.zip` is released on **HuggingFace**, not S3 — symmetric with the
detector (which already lives on HF and is pulled by `fetch_detector`). So *all*
model artifacts come from HuggingFace, and the release pipeline needs **no AWS**.

- **One HF model repo:** `pyronear/temporal-model` (configurable; a `RELEASE_REPO`
  default constant), holding a single `model.zip`.
- **One version = one HF git revision/tag:** each release is committed to the repo
  and tagged `v<version>`. `fetch` pins the exact bytes with
  `revision="v<version>"`. This mirrors "one git tag per version" — the source tag
  and the HF tag share the name.
- **Public repo** — so the CI `fetch` needs **no token**. Only `publish` needs a
  write token (the maintainer's, used locally — never in CI).

> The S3 bucket `pyronear-temporal-model` created earlier is **no longer used** for
> model release and can be deleted. (DVC's `pyro-vision-rd` S3 remote is unrelated
> and stays — it backs the pipeline, not the release.)

## Background

Established facts (current repo state):

- `push.yml` (merged) already builds the API image and pushes it to **Docker Hub**
  (`pyronear/temporal-model-api`) on pushes to `main` (tag `latest`) and on `v*`
  tags (tag = version). It builds a **lean** image — the model is *not* baked in;
  `.dockerignore` excludes `api/models` and `**/data`, and `MODEL_PATH` defaults to
  `/models/model.zip` (a runtime mount).
- The model-versioning spec decided the deployable should be **coupled** (model
  baked into the image, image tag = model version). This spec reconciles the
  workflow with that decision.
- `huggingface_hub` is already a `core` dependency (used by `fetch_detector`); the
  release CLI reuses it. A local `api/models/model.zip` (163 MB) exists but is a
  legacy package whose manifest has **no `model_version`** (it predates stamping).

## Decisions

| Decision | Choice |
|---|---|
| Image shape | Model **baked in** — self-contained, immutable, image tag = version. |
| Artifact home | **HuggingFace** repo `pyronear/temporal-model`; one `model.zip`, one revision/tag `v<version>` per release. No S3. |
| Model source (build) | `fetch` via `hf_hub_download(repo, "model.zip", revision="v<version>")`; assert `manifest.model_version == <version>`. |
| Put artifacts on HF | `publish` CLI: **stamp** `model_version = <version>` into the manifest, upload `model.zip`, tag the HF revision `v<version>`. Version declared once, at publish (matching the tag) — not baked in at packaging. |
| Auth | **Public HF repo** → CI `fetch` needs **no token**. `publish` uses the maintainer's HF write token locally (never in CI). No AWS. |
| Registry | Docker Hub `pyronear/temporal-model-api` (unchanged). |
| Trigger | `v*` tags **only** — drop the `main`→`latest` build (no ambiguous "latest" model). |
| Orchestration | `scripts/release-api.sh` + adapted `push.yml`, both calling the `release` CLI. |
| Logic home | `temporal_model.api.release` (`huggingface_hub`-based), runnable locally and in CI. |
| First release | Bootstrap `0.1.0` by publishing the legacy `model.zip` (publish stamps the version). |
| Packaging stage | Out of scope — implemented as the `train` DVC `package` stage. Produces a provenance-stamped but **version-less** `model.zip`; the version is applied at publish. Precondition: a `model.zip` has been **published** to the HF repo at `v<version>`. |

## Components

### `temporal_model.api.release` — the testable core

A new module/CLI in the `api` package with two subcommands, built on
`huggingface_hub` (add `huggingface-hub` to `api`'s deps, or rely on the `core`
dependency) and a shared manifest reader (zip → `manifest.yaml` → `model_version`).
It is independent of the running API's frames-S3 settings — the runtime never
fetches the model (it's baked in).

- **`fetch --version X.Y.Z --output <path> [--repo R]`**
  `hf_hub_download(repo_id=R, filename="model.zip", revision="vX.Y.Z")`, reads the
  manifest, **asserts `model_version == X.Y.Z`** (error if missing or mismatched),
  writes the zip to `<path>`. Used by the CI build to place the model at
  `api/models/model.zip`.
- **`publish --version X.Y.Z --file <model.zip> [--repo R]`**
  **Stamps** `model_version = X.Y.Z` into the file's manifest (in place), then
  `HfApi.upload_file(path_in_repo="model.zip", …)` and `create_tag(R,
  tag="vX.Y.Z")`. This is where the version is *applied* — the artifact from
  packaging has provenance but no `model_version`. Used to put artifacts *onto* HF
  (bootstrap now; the packaging stage later).

The **git tag is the single source of truth** for the version: `publish` stamps it
into the manifest and names the HF revision after it, and `fetch` re-checks it at
build time. So one human decision (the version passed to `publish`, mirrored by the
tag) flows into the manifest, the HF revision, and the image tag.

### Dockerfile + `.dockerignore` — bake the model

- `.dockerignore`: stop excluding the baked model. Replace the blanket
  `api/models` ignore with a rule that still ignores the directory **except**
  `api/models/model.zip`:
  ```
  api/models/*
  !api/models/model.zip
  ```
- `api/Dockerfile`: after the app is set up, `COPY api/models/model.zip
  /models/model.zip`. `MODEL_PATH` already defaults to `/models/model.zip`, so no
  app change is needed. The running container needs **no model fetch**.

### `scripts/release-api.sh` — orchestration (local + CI)

```bash
scripts/release-api.sh <version>
```
1. `python -m temporal_model.api.release fetch --version <version> --output api/models/model.zip`
2. `docker build -f api/Dockerfile . -t pyronear/temporal-model-api:<version>`
3. (push is left to the caller — CI logs in and pushes; locally you stop after build)

Locally you run this with your HF token to validate fetch + assert + build end to
end **before** any tag is pushed.

### `push.yml` — adapted workflow

On `v*` tags:
1. Resolve `VERSION=${GITHUB_REF#refs/tags/v}`.
2. `python -m temporal_model.api.release fetch --version $VERSION --output api/models/model.zip`
   (public repo — no token needed).
3. `docker build -f api/Dockerfile . -t pyronear/temporal-model-api:$VERSION`.
4. Docker Hub login (existing `DOCKERHUB_*` secrets) → `docker push …:$VERSION`.

The `main`→`latest` branch trigger and the `latest` tag fallback are removed.

### Bootstrap — the first artifact (`0.1.0`)

One-time, run locally with your HF write token (the packaging stage that would
normally produce the artifact is deferred). Because `publish` stamps the version,
the legacy zip needs no separate stamping step:

1. **Publish**: `python -m temporal_model.api.release publish --version 0.1.0
   --file api/models/model.zip` → stamps `model_version: "0.1.0"` into the manifest,
   uploads `model.zip` to `pyronear/temporal-model`, and tags the revision `v0.1.0`.
   (The legacy package gains `model_version` but no `provenance` block — provenance
   comes from the `train` packaging stage for `0.2.0+`.)
2. **Tag**: `git tag v0.1.0 && git push origin v0.1.0` → the workflow builds and
   pushes `pyronear/temporal-model-api:0.1.0`.

## Release flow

```
(precondition) model.zip is published to HF repo pyronear/temporal-model @ tag v<version>

push git tag vX.Y.Z
  └─► push.yml:
        1. VERSION = X.Y.Z
        2. release fetch --version X.Y.Z --output api/models/model.zip   (hf_hub_download @ revision + assert)
        3. docker build -f api/Dockerfile .  -t pyronear/temporal-model-api:X.Y.Z   (bakes /models/model.zip)
        4. docker push pyronear/temporal-model-api:X.Y.Z

deploy:   run pyronear/temporal-model-api:X.Y.Z   (no model mount, no model fetch)
rollback: redeploy the previous tag
```

## Releasing a new version (worked example: `0.2.0`)

`0.1.0` was a one-off bootstrap of a *legacy* zip; `0.2.0` is the steady-state path.
Assume `0.2.0` is a retrained/updated model.

1. **Produce the artifact — provenance-stamped, version-less.** The model is
   packaged by the **`train` DVC pipeline** (a stage that calls
   `core.build_model_package(..., train_git_sha=<sha>)`), which writes the
   `provenance` block (backbone + `detector` from `core/detector.yaml`) but leaves
   `model_version` unset. The artifact's identity at this point is its content hash
   + provenance; the human version is applied at publish. *(Run it with
   `dvc repro package` in `train/`.)*
2. **Publish to HF — this applies the version:**
   `python -m temporal_model.api.release publish --version 0.2.0 --file model.zip`
   → stamps `model_version: "0.2.0"` into the manifest, uploads `model.zip`, tags
   the HF revision `v0.2.0`.
3. **Tag:** `git tag v0.2.0 && git push origin v0.2.0` — the only trigger.
4. **CI (unchanged):** resolves `0.2.0` → `fetch` (download @ revision + assert) →
   `docker build` (bakes the model) → push `pyronear/temporal-model-api:0.2.0`.
5. **Deploy/rollback:** run `:0.2.0`; roll back to `:0.1.0`.

Notable:
- **No code or workflow changes** to ship `0.2.0` — releasing a new model is pure
  *data*: a new HF revision + a new git tag.
- **Detector:** if `0.2.0` uses a new YOLO, that's a *prior* change (bump
  `detector.yaml` + `fetch_detector` + dvc-track); it flows into `provenance.detector`
  at packaging. Unchanged detector → nothing to do.
- **One-version semantics:** if `0.2.0` were an API-code-only change with the same
  model, you still publish a `model.zip` at HF revision `v0.2.0` — re-`publish` the
  same bytes stamped as `0.2.0`. The version names the whole deployable, so the HF
  revision always tracks it.

## Secrets

| Secret | Purpose |
|---|---|
| `DOCKERHUB_LOGIN` / `DOCKERHUB_PW` | Docker Hub push (existing). |

The HF repo is **public**, so the CI `fetch` needs **no secret**. Publishing uses
the maintainer's own HF write token locally — CI never needs HF access. **No AWS
secrets, no HF secret in CI.**

## Testing

- **`release.fetch` / `release.publish`** — unit-tested by **mocking
  `huggingface_hub`** (`hf_hub_download`, `HfApi.upload_file`, `create_tag`) —
  mirroring the `fetch_detector` test pattern (no network):
  - `fetch` success (stamped manifest matches version, correct `revision` passed),
    `model_version` mismatch (error), missing `model_version` (error).
  - `publish` success: stamps `model_version` into a **temp copy** (the caller's
    file is left untouched), uploads it, and tags `v<version>`. Versions are
    **immutable** — re-publishing an existing version fails at `create_tag` (no
    silent overwrite).
- **No GPU / network** — HF calls mocked; the manifest reader/stamper works on a
  tiny fixture zip.
- **Docker build** is validated manually/locally (and by the first CI run); not
  unit-tested.

## Non-goals (this spec)

- The **packaging stage** that produces the `model.zip` from a trained checkpoint —
  the `train` DVC `package` stage (already implemented). It outputs a
  provenance-stamped but version-less zip; `publish` applies the version.
  Precondition only: a `model.zip` has been published to the HF repo at `v<version>`.
- A model card / rich HF repo metadata (a plain repo with `model.zip` is enough;
  a card can be added later).
- A GitHub Release object with notes/provenance (the versioning spec's "index" —
  can be added to `push.yml` later; not required to ship images).
- Per-version image retention/cleanup; multi-arch images.
- Adding a `provenance` block to the legacy bootstrap package (publish only stamps
  `model_version`; provenance arrives with the `train` packaging stage for `0.2.0+`).

## Success criteria

1. `release publish` **stamps** `model_version = <version>` into the manifest,
   uploads `model.zip`, and tags the HF revision `v<version>`; `release fetch`
   downloads that revision and **asserts** `manifest.model_version == <version>`.
   Unit tests cover stamping a version-less input and fetch on
   matching/missing/wrong-version, with HF mocked.
2. The Dockerfile bakes `api/models/model.zip` to `/models/model.zip`; the image
   runs with **no** model fetch and **no** model mount.
3. `scripts/release-api.sh <version>` runs the fetch + build locally with the
   maintainer's HF token.
4. `push.yml` on a `v*` tag produces and pushes
   `pyronear/temporal-model-api:<version>` with the matching model baked in.
5. The bootstrap publishes `0.1.0` to HF (revision `v0.1.0`) and a `v0.1.0` git tag
   yields a working baked image.
6. Rollback is "redeploy the previous tag," no other change.
