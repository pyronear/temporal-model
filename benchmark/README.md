# temporal-model-benchmark

Measures the temporal smoke classifier's latency, throughput, and resource
usage, with a per-stage breakdown of `predict()`. Import as
`temporal_model.benchmark`. Depends on `temporal-model-core`.

Phase 1 covers the **core in-process** path. See
`docs/specs/2026-06-09-benchmark-package-design.md`.

## Run

```bash
make install                       # uv sync (installs core editable)
temporal-benchmark core \
    --store data/sequences \
    --model ../api/models/model.zip \
    --out results
```

Outputs a self-describing dir `results/<host>-<timestamp>/` with `raw.parquet`,
`resources.parquet`, `summary.json`, `plots/*.png`, and `report.md`.

## On a VM

See `scripts/provision_vm.sh`, `scripts/push_data.sh`, `scripts/pull_results.sh`.
