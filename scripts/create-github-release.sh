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
