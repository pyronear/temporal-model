#!/usr/bin/env bash
# Fetch the released model.zip from HuggingFace and build the API image.
# Usage: scripts/release-api.sh <version>   (run from repo root)
set -euo pipefail

VERSION="${1:?usage: release-api.sh <version>}"
IMAGE="pyronear/temporal-model-api:${VERSION}"

(cd api && uv run python -m temporal_model.api.release \
    fetch --version "$VERSION" --output models/model.zip)

docker build -f api/Dockerfile . -t "$IMAGE"
echo "built $IMAGE (model.zip baked from HF v$VERSION)"
echo "push with: docker push $IMAGE"
