#!/usr/bin/env bash
# Stand up the API + MinIO compose stack with profiling on, on a VM.
# Usage: provision_api_vm.sh <ssh-host>
# Assumes the repo is already on the VM at ~/temporal-model with api/models/model.zip.
set -euo pipefail
HOST="${1:?usage: provision_api_vm.sh <ssh-host>}"

ssh "$HOST" 'command -v docker >/dev/null || (curl -fsSL https://get.docker.com | sudo sh)'
# Bring the stack up with profiling enabled (detached). `sudo env VAR=...`
# sets the var directly in the command environment docker compose reads, so
# `${TEMPORAL_API_PROFILE}` interpolation works regardless of sudoers env policy.
ssh "$HOST" 'cd temporal-model/api && \
    sudo env TEMPORAL_API_PROFILE=true docker compose up -d --build'
# Wait for the API to load the model and report ready before returning.
echo "waiting for API /health..."
ssh "$HOST" 'for i in $(seq 1 60); do \
    curl -sf http://localhost:8000/health 2>/dev/null | grep -q "\"model_loaded\":true" \
        && { echo "API ready"; exit 0; }; \
    sleep 3; done; echo "WARNING: API not ready after 180s"; exit 1'
echo "stack up on $HOST — API on :8000, MinIO on :9000. Next: upload_frames_to_minio.py"
