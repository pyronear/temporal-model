#!/usr/bin/env bash
# rsync the local pyro-annotator sequence store to the VM (no S3 creds needed there).
# Run from the benchmark/ dir. Usage: push_data.sh <ssh-host>
set -euo pipefail
HOST="${1:?usage: push_data.sh <ssh-host>}"
SRC="data/sequences/pyro-annotator/"
DST="$HOST:~/temporal-model/benchmark/data/sequences/pyro-annotator/"
test -d "$SRC" || { echo "missing $SRC (run 'dvc pull' first)"; exit 1; }
rsync -az --info=progress2 "$SRC" "$DST"
echo "pushed dataset to $HOST"
