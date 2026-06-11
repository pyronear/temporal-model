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

- A `scripts/create-github-release.sh` script (with a stub-`gh` contract test)
  holding the header + `gh release create` logic.
- A new `release` job in `.github/workflows/push.yml` that calls the script on
  every `vX.Y.Z` tag push.
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
2. Run `scripts/create-github-release.sh "$GITHUB_REF_NAME"` with
   `GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}` in the step env.

The logic lives in a script, not inline YAML, so it is testable (see
Verification) and mirrors the existing `scripts/release-api.sh`.

### Script — `scripts/create-github-release.sh`

Takes the tag (`vX.Y.Z`) as `$1` and:

1. Derives `VERSION="${TAG#v}"`.
2. Reads `MODEL_VERSION=$(cat api/MODEL_VERSION)`; exits non-zero if empty
   (matching the `docker` job's guard).
3. Writes a notes-header to a temp file:

   ```
   **Docker image:** `pyronear/temporal-model-api:<VERSION>`
   **Bundled model:** v<MODEL_VERSION> (api/MODEL_VERSION)
   ```

4. Runs:

   ```
   gh release create "$TAG" --notes-file <header-file> --generate-notes
   ```

`--generate-notes` makes GitHub append its auto-generated "What's Changed"
(merged PRs / commits since the previous tag) below the header. Auth uses the
built-in `GITHUB_TOKEN`, read from `GH_TOKEN` in the environment. The script is
`set -euo pipefail` and runs from the repo root (so `api/MODEL_VERSION` resolves).

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

The `release` job only fires on real tag pushes, and a real run has outward side
effects (public image push + public GitHub Release), so it is not exercised as
routine verification. Coverage is layered:

- **Script contract test** (`scripts/test-create-github-release.sh` or a bats
  test) — put a stub `gh` on `PATH` that records its arguments, run the script
  with sample `TAG` / a temp `api/MODEL_VERSION`, and assert: (a) the header file
  content (image tag + bundled model line), and (b) the exact `gh release create`
  args (`<tag>`, `--notes-file`, `--generate-notes`). Also assert it exits
  non-zero when `MODEL_VERSION` is empty. This is the on-demand pass/fail check
  for the script's behavior.
- **`actionlint`** on `push.yml` — confirms the job graph (`docker` → `release`),
  per-job permissions (`contents: write` on `release`), and the `GH_TOKEN` env
  wiring parse correctly.
- **Live GitHub integration** (Release actually created, "What's Changed" notes
  generated) — this is GitHub's behavior, not ours, and is confirmed by observing
  the next genuine release tag. No throwaway test tag is pushed.

## Files touched

- `scripts/create-github-release.sh` — new; header + `gh release create` logic.
- `scripts/test-create-github-release.sh` — new; stub-`gh` contract test.
- `.github/workflows/push.yml` — add the `release` job (calls the script) and
  per-job permissions.
- `docs/releasing.md` — new release runbook.
