#!/usr/bin/env bash
# Stand up the API + MinIO compose stack with profiling on, on a VM.
# Usage: provision_api_vm.sh <ssh-host>
# Assumes the repo is already on the VM at ~/temporal-model with api/models/model.zip.
set -euo pipefail
HOST="${1:?usage: provision_api_vm.sh <ssh-host>}"

ssh "$HOST" 'command -v docker >/dev/null || (curl -fsSL https://get.docker.com | sudo sh)'
ssh "$HOST" 'sudo usermod -aG docker "$USER" || true'
# Bring the stack up with profiling enabled (detached).
ssh "$HOST" 'cd temporal-model/api && \
    TEMPORAL_API_PROFILE=true sudo -E docker compose up -d --build'
echo "stack up on $HOST — API on :8000, MinIO on :9000. Next: upload_frames_to_minio.py"
