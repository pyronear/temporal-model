# Auto-create the GitHub Release on tag push

**Date:** 2026-06-11
**Status:** Approved, ready for implementation plan

## Problem

Cutting a release today is partly automated, partly manual:

- Pushing a `vX.Y.Z` git tag triggers `.github/workflows/push.yml`, which builds
  and pushes the Docker image `pyronear/temporal-model-api:<version>` (and
  `:latest`), baking in the model pinned by `api/MODEL_VERSION`.
- The **GitHub Release** object (the entry on the repo's Releases page, with
  notes) is created **by hand** after the fact. Nothing in CI produces it.

We want the GitHub Release to be created automatically when a tag is pushed, with
generated notes plus a small header pointing consumers at the image and bundled
model. And the whole release flow should be documented so it is repeatable.

## Scope

In scope:

- A new `release` job in `.github/workflows/push.yml` that creates the GitHub
  Release on every `vX.Y.Z` tag push.
- A `docs/releasing.md` runbook documenting the end-to-end release flow.

Explicitly out of scope (unchanged):

- The model publish / HuggingFace flow (`release.py publish`, `api/MODEL_VERSION`).
- The Docker build/push (`docker` job stays as-is).
- No `CHANGELOG.md`, no `pyproject.toml` version bumps (code version is injected
  from the git tag at build time; pyproject is not the source of truth).
- No pre-release / release-candidate handling — every tag is a full release.

## Versioning context (the thing that's easy to get wrong)

Two version axes are **decoupled**:

- **Code / API version** — the git tag `vX.Y.Z`. Drives the image tag. Bumps
  whenever serving code changes, even with no retraining.
- **Model version** — pinned in `api/MODEL_VERSION`, published to the HuggingFace
  model repo as its own `v<version>` tag. Bumps only when the bundled model
  changes.

A release can bump the code version without touching the model version (and the
header in the release notes makes the pairing explicit).

## Design

### Workflow change — `.github/workflows/push.yml`

Add a `release` job:

- `needs: docker` — runs only after the image is built and pushed, so a Release
  never points at an image that does not exist.
- Permissions scoped per job: the existing `docker` job keeps `contents: read`;
  the new `release` job gets `contents: write` (required to create a Release).
  The top-level `permissions:` is set so each job grants only what it needs.

Steps in the `release` job:

1. `actions/checkout@v6` at the tagged commit.
2. Resolve versions (same logic the `docker` job already uses):
   - `VERSION=${GITHUB_REF#refs/tags/v}`
   - `MODEL_VERSION=$(cat api/MODEL_VERSION)` (fail if empty, matching `docker`).
3. Build a notes-header file:

   ```
   **Docker image:** `pyronear/temporal-model-api:<VERSION>`
   **Bundled model:** v<MODEL_VERSION> (api/MODEL_VERSION)
   ```

4. Create the Release:

   ```
   gh release create "$GITHUB_REF_NAME" \
     --notes-file <header-file> \
     --generate-notes
   ```

   `--generate-notes` makes GitHub append its auto-generated "What's Changed"
   (merged PRs / commits since the previous tag) below the header. Auth uses the
   built-in `GITHUB_TOKEN` (`env: GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`).

### Behavior

- Every `vX.Y.Z` tag → exactly one full GitHub Release.
- `gh release create` **fails if a Release already exists** for that tag. This is
  the desired guard: no silent overwrite, consistent with the immutability the
  model publish already enforces on HF tags.
- The job adds no new secrets; `GITHUB_TOKEN` is provided automatically.

### Documentation — `docs/releasing.md`

A runbook covering the full flow:

1. The two decoupled version axes (code vs model) and when each bumps.
2. When the bundled model changes: bump `api/MODEL_VERSION`, publish the
   `model.zip` to HuggingFace via `release.py publish`.
3. Cut the release: push the `vX.Y.Z` git tag. CI then:
   - builds + pushes `pyronear/temporal-model-api:<version>` and `:latest`;
   - auto-creates the GitHub Release with the header + auto-generated notes.
4. What the maintainer no longer does by hand: writing the GitHub Release notes.

## Verification

This workflow only fires on real tag pushes, so the genuine end-to-end check is
the next tag. Before that:

- Lint the workflow YAML (`actionlint` if available) and confirm the job graph
  (`docker` → `release`) and per-job permissions parse correctly.
- Dry-run the notes-header construction locally (shell snippet producing the
  header file from a sample `VERSION` / `MODEL_VERSION`).
- Eyeball that `GH_TOKEN` / `permissions: contents: write` are wired on the
  `release` job.

## Files touched

- `.github/workflows/push.yml` — add the `release` job and per-job permissions.
- `docs/releasing.md` — new release runbook.
