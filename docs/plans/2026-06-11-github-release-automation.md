# GitHub Release Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-create the GitHub Release on every `vX.Y.Z` tag push, with a header (image tag + bundled model version) followed by GitHub's auto-generated "What's Changed" notes.

**Architecture:** A small `scripts/create-github-release.sh` holds the logic (derive version, read `api/MODEL_VERSION`, build header, call `gh release create --generate-notes`); the existing `.github/workflows/push.yml` gains a `release` job that calls it after the image is pushed. A plain-bash contract test stubs `gh` to verify the script's header content and CLI arguments without any outward side effects.

**Tech Stack:** Bash, GitHub Actions, `gh` CLI (`GITHUB_TOKEN`). No new dependencies; no `bats`/`actionlint`/`shellcheck` available, so tests are plain bash and YAML is validated with Python's `yaml` loader.

Spec: `docs/specs/2026-06-11-github-release-automation-design.md`

---

## File Structure

- `scripts/create-github-release.sh` — **new.** The release logic. Mirrors the existing `scripts/release-api.sh` (same shebang, `set -euo pipefail`, `$1` arg, run-from-repo-root assumption).
- `scripts/test-create-github-release.sh` — **new.** Self-contained bash contract test: stubs `gh` on `PATH`, runs the script in a sandbox dir, asserts header + args. Run directly (`bash scripts/test-create-github-release.sh`).
- `.github/workflows/push.yml` — **modify.** Add a `release` job (`needs: docker`, `permissions: contents: write`) that checks out and runs the script.
- `docs/releasing.md` — **new.** End-to-end release runbook.

---

## Task 1: The release script + contract test (TDD)

**Files:**
- Create: `scripts/create-github-release.sh`
- Test: `scripts/test-create-github-release.sh`

The test is written first and must fail (script absent) before the script is implemented.

- [ ] **Step 1: Write the failing contract test**

Create `scripts/test-create-github-release.sh` with exactly this content:

```bash
#!/usr/bin/env bash
# Contract test for create-github-release.sh.
#
# Stubs the `gh` CLI on PATH (no network, no real Release) and runs the script
# in a sandbox dir holding a fake api/MODEL_VERSION, then asserts the header
# content and the exact `gh release create` arguments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/create-github-release.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- stub gh: record argv and copy the --notes-file content ---
mkdir -p "$WORK/bin"
cat > "$WORK/bin/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" > "$GH_ARGS_FILE"
prev=""
for a in "$@"; do
  [ "$prev" = "--notes-file" ] && cp "$a" "$GH_NOTES_FILE"
  prev="$a"
done
STUB
chmod +x "$WORK/bin/gh"

export PATH="$WORK/bin:$PATH"
export GH_ARGS_FILE="$WORK/args.txt"
export GH_NOTES_FILE="$WORK/notes.txt"

fail() { echo "FAIL: $1"; exit 1; }

# --- case 1: happy path ---
mkdir -p "$WORK/repo/api"
echo "9.9.9" > "$WORK/repo/api/MODEL_VERSION"
( cd "$WORK/repo" && bash "$SCRIPT" v1.2.3 ) || fail "script exited non-zero on happy path"

args="$(cat "$GH_ARGS_FILE")"
echo "$args" | grep -q "release create v1.2.3" || fail "missing 'release create v1.2.3' in: $args"
echo "$args" | grep -q -- "--notes-file" || fail "missing --notes-file in: $args"
echo "$args" | grep -q -- "--generate-notes" || fail "missing --generate-notes in: $args"

notes="$(cat "$GH_NOTES_FILE")"
echo "$notes" | grep -qF 'pyronear/temporal-model-api:1.2.3' || fail "missing image tag in header: $notes"
echo "$notes" | grep -qF 'v9.9.9 (api/MODEL_VERSION)' || fail "missing bundled-model line in header: $notes"

# --- case 2: empty MODEL_VERSION must abort before calling gh ---
rm -f "$GH_ARGS_FILE"
: > "$WORK/repo/api/MODEL_VERSION"
if ( cd "$WORK/repo" && bash "$SCRIPT" v1.2.3 ) 2>/dev/null; then
  fail "script should exit non-zero when MODEL_VERSION is empty"
fi
[ -f "$GH_ARGS_FILE" ] && fail "gh was called despite empty MODEL_VERSION"

echo "PASS"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
chmod +x scripts/test-create-github-release.sh
bash scripts/test-create-github-release.sh
```
Expected: non-zero exit / `FAIL` (the script `scripts/create-github-release.sh` does not exist yet, so `bash "$SCRIPT"` fails with "No such file or directory").

- [ ] **Step 3: Write the script**

Create `scripts/create-github-release.sh` with exactly this content:

```bash
#!/usr/bin/env bash
# Create the GitHub Release for a pushed tag.
#
# Body = a header (image tag + bundled model version) followed by GitHub's
# auto-generated "What's Changed" notes. Run from the repo root by the `release`
# job in .github/workflows/push.yml; needs GH_TOKEN in the environment.
# Usage: scripts/create-github-release.sh <tag>   (e.g. v0.3.1)
set -euo pipefail

TAG="${1:?usage: create-github-release.sh <tag>}"
VERSION="${TAG#v}"

MODEL_VERSION="$(cat api/MODEL_VERSION)"
if [ -z "$MODEL_VERSION" ]; then
    echo "api/MODEL_VERSION is missing or empty" >&2
    exit 1
fi

NOTES_HEADER="$(mktemp)"
trap 'rm -f "$NOTES_HEADER"' EXIT
cat > "$NOTES_HEADER" <<EOF
**Docker image:** \`pyronear/temporal-model-api:${VERSION}\`
**Bundled model:** v${MODEL_VERSION} (api/MODEL_VERSION)
EOF

gh release create "$TAG" --notes-file "$NOTES_HEADER" --generate-notes
echo "created GitHub Release $TAG (image $VERSION, model $MODEL_VERSION)"
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
chmod +x scripts/create-github-release.sh
bash scripts/test-create-github-release.sh
```
Expected: `PASS` and exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/create-github-release.sh scripts/test-create-github-release.sh
git commit -m "feat(release): script to create the GitHub Release on tag push"
```

---

## Task 2: Wire the `release` job into the workflow

**Files:**
- Modify: `.github/workflows/push.yml`

The current file has top-level `permissions: contents: read` and a single `docker` job. Add a `release` job that runs after it.

- [ ] **Step 1: Add the `release` job**

Append this job to `.github/workflows/push.yml`, indented as a sibling of `docker:` under `jobs:` (two-space indent for the job key, matching `docker:`):

```yaml
  release:
    needs: docker
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v6
      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: scripts/create-github-release.sh "$GITHUB_REF_NAME"
```

Leave the top-level `permissions: contents: read` as-is — it still applies to `docker`; the job-level `permissions: contents: write` here overrides it for `release` only.

- [ ] **Step 2: Validate the workflow YAML parses**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/push.yml'))" && echo "YAML OK"
```
Expected: `YAML OK` (no traceback).

- [ ] **Step 3: Verify the job graph and permissions structurally**

Run:
```bash
python3 - <<'PY'
import yaml
wf = yaml.safe_load(open(".github/workflows/push.yml"))
jobs = wf["jobs"]
assert "release" in jobs, "release job missing"
rel = jobs["release"]
assert rel["needs"] == "docker", f"release.needs should be docker, got {rel['needs']!r}"
assert rel["permissions"]["contents"] == "write", "release needs contents: write"
assert jobs["docker"].get("permissions", wf.get("permissions")) is not None
steps = rel["steps"]
assert any("create-github-release.sh" in (s.get("run") or "") for s in steps), "script call missing"
assert any((s.get("env") or {}).get("GH_TOKEN") for s in steps), "GH_TOKEN env missing"
print("job graph OK")
PY
```
Expected: `job graph OK`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/push.yml
git commit -m "feat(ci): auto-create the GitHub Release on tag push"
```

---

## Task 3: Release runbook

**Files:**
- Create: `docs/releasing.md`

- [ ] **Step 1: Write the runbook**

Create `docs/releasing.md` with exactly this content:

````markdown
# Releasing

This repo has **two decoupled version axes**. Get the distinction right and the
rest is mechanical.

| Axis | Source of truth | Where it lands |
|------|-----------------|----------------|
| **Code / API version** | the git tag `vX.Y.Z` | Docker image `pyronear/temporal-model-api:<version>` (+ `:latest`); reported by `/health`, `/predict`, OpenAPI |
| **Model version** | `api/MODEL_VERSION` | the HuggingFace model repo, tagged `v<version>`; baked into the image at build time |

The code version bumps whenever serving code changes — **even with no
retraining**. The model version bumps **only** when the bundled model changes.
A release can move the code version while leaving the model version untouched.

## Cutting a release

### 1. (Only if the model changed) publish the new model

Bump the pin and publish `model.zip` to HuggingFace (needs a **write** HF token):

```bash
echo "X.Y.Z" > api/MODEL_VERSION   # the new model version
cd api && uv run python -m temporal_model.api.release \
    publish --version X.Y.Z --file path/to/model.zip
```

`publish` stamps the version into the manifest, uploads the zip + model card,
and tags `vX.Y.Z` on the HF repo (immutable — re-publishing an existing version
fails). Commit the `api/MODEL_VERSION` change.

### 2. Push the git tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

That triggers `.github/workflows/push.yml`, which:

1. fetches the `model.zip` pinned by `api/MODEL_VERSION` from HuggingFace,
2. builds and pushes `pyronear/temporal-model-api:X.Y.Z` and `:latest`,
3. **auto-creates the GitHub Release** for `vX.Y.Z` — a header naming the image
   and bundled model, followed by GitHub's auto-generated "What's Changed".

The maintainer no longer writes release notes by hand. `gh release create` fails
if a Release for the tag already exists, so tags are effectively immutable.

## Notes

- The GitHub Release logic lives in `scripts/create-github-release.sh`; its
  behavior is covered by `scripts/test-create-github-release.sh`.
- `pyproject.toml` versions are not the source of truth for the released version
  — the code version is injected from the git tag at build time.
````

- [ ] **Step 2: Verify it renders / has no broken structure**

Run:
```bash
python3 -c "open('docs/releasing.md').read(); print('docs/releasing.md OK')"
```
Expected: `docs/releasing.md OK`.

- [ ] **Step 3: Commit**

```bash
git add docs/releasing.md
git commit -m "docs: add the release runbook"
```

---

## Final verification (whole feature)

- [ ] **Run the contract test once more**

```bash
bash scripts/test-create-github-release.sh
```
Expected: `PASS`.

- [ ] **Re-validate the workflow**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/push.yml'))" && echo OK
```
Expected: `OK`.

- [ ] **Confirm the spec's verification layers are satisfied**

Script contract test green (header + args), workflow YAML + job graph valid. The
live GitHub integration ("What's Changed" notes actually generated, Release
actually created) is confirmed on the **next real release tag** — no throwaway
test tag is pushed, because that has the same outward side effects as a real
release.

---

## Notes for the implementer

- **Heredoc backticks:** in `create-github-release.sh` the header uses `` \` `` (escaped backticks) inside an unquoted `<<EOF` heredoc so `${VERSION}` expands but the backticks stay literal. Keep them escaped.
- **`GITHUB_REF_NAME`** on a tag push is the tag itself (e.g. `v0.3.1`) — that is what the script receives and strips `v` from.
- Do not touch the `docker` job, `release.py`, `api/MODEL_VERSION`, or `pyproject.toml` versions — all out of scope.
