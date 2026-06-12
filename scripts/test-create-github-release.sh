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
