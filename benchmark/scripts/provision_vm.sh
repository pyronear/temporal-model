#!/usr/bin/env bash
# Bootstrap a fresh VM to run the core benchmark: uv, repo, python 3.12, deps, model.
# Usage: provision_vm.sh <ssh-host> [repo-url]
set -euo pipefail
HOST="${1:?usage: provision_vm.sh <ssh-host> [repo-url]}"
REPO="${2:-https://github.com/pyronear/temporal-model.git}"

ssh "$HOST" "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
ssh "$HOST" "test -d temporal-model || git clone '$REPO' temporal-model"
ssh "$HOST" "cd temporal-model && \$HOME/.local/bin/uv python install 3.12 && \
             make -C benchmark install && make fetch-model"
echo "provisioned $HOST — now run scripts/push_data.sh $HOST"
