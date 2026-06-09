#!/usr/bin/env bash
# rsync benchmark results back from the VM. Run from benchmark/. Usage: pull_results.sh <ssh-host>
set -euo pipefail
HOST="${1:?usage: pull_results.sh <ssh-host>}"
rsync -az --info=progress2 \
    "$HOST:~/temporal-model/benchmark/results/" "./results/"
echo "pulled results from $HOST into ./results/"
