# temporal-model-benchmark

Measures the temporal smoke classifier's latency, throughput, and resource
usage, with a per-stage breakdown of `predict()`. Import as
`temporal_model.benchmark`. Depends on `temporal-model-core`.

Phase 1 covers the **core in-process** path. See
`docs/specs/2026-06-09-benchmark-package-design.md`.

## Run

```bash
make install                       # uv sync (installs core editable)
dvc pull                           # fetch the pyro-annotator store into data/03_primary
temporal-benchmark core \
    --model ../api/models/model.zip
```

Data follows the Kedro-style layers used across this repo: the input sequence
store lives in `data/03_primary/sequences/` (DVC-tracked) and each run writes a
self-describing dir `data/08_reporting/<host>-<timestamp>/` with `raw.parquet`,
`resources.parquet`, `summary.json`, `plots/*.png`, and `report.md`.

## On a VM

See `scripts/provision_vm.sh`, `scripts/push_data.sh`, `scripts/pull_results.sh`.
