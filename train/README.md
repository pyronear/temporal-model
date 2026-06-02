# temporal-model-train

DVC training pipeline for the temporal smoke classifier.

Import as `temporal_model.train`; CLI entry point `temporal-train`. Depends on
`temporal-model-core`. Scaffold stage — `dvc.yaml` holds a placeholder stage and
`train.py` is a stub.

```bash
make install
uv run dvc repro       # once real stages exist
```
