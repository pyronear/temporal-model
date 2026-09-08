# Releasing

This repo has **two decoupled version axes**. Get the distinction right and the
rest is mechanical.

| Axis | Source of truth | Where it lands |
|------|-----------------|----------------|
| **Code / API version** | the git tag `vX.Y.Z` | Docker image `pyronear/temporal-model-api:<version>` (+ `:latest`); reported by `/health`, `/predict`, OpenAPI |
| **Model version** | `api/MODEL_VERSION` | the HuggingFace model repo, tagged `v<version>`; baked into the image at build time |

The code version bumps whenever serving code changes — **even with no
retraining**. The model version bumps **only** when the bundled model changes.
A release can move the code version while leaving the model version untouched.

## Cutting a release

### 1. (Only if the model changed) publish the new model

Bump the pin and publish `model.zip` to HuggingFace (needs a **write** HF token):

```bash
echo "X.Y.Z" > api/MODEL_VERSION   # the new model version
cd api && uv run python -m temporal_model.api.release \
    publish --version X.Y.Z --file path/to/model.zip
```

`publish` stamps the version into the manifest, uploads the zip + model card,
and tags `vX.Y.Z` on the HF repo (immutable — re-publishing an existing version
fails). Commit the `api/MODEL_VERSION` change.

Ship the torch-free runtime artifact with it: export `model_onnx.zip` from the
same `model.zip` (the export checks torch/ONNX parity and records the source
SHA-256 in its manifest) and pass it to `publish` so both land under one tag:

```bash
# from the repo root; MODEL=/abs/path/to/model.zip, ONNX=/abs/path/to/model_onnx.zip
uv run --project core temporal-export-onnx --model "$MODEL" --output "$ONNX"
uv run --project api python -m temporal_model.api.release \
    publish --version X.Y.Z --file "$MODEL" --onnx-file "$ONNX"
```

`publish` refuses an `--onnx-file` whose manifest was not exported from that
exact `model.zip`, then re-stamps its source hash from the uploaded (version
stamped) copy. Consumers fetch it with `release fetch --onnx` (or
`make fetch-model-onnx`).

### 2. Push the git tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

That triggers `.github/workflows/push.yml`, which:

1. fetches the `model.zip` pinned by `api/MODEL_VERSION` from HuggingFace,
2. builds and pushes `pyronear/temporal-model-api:X.Y.Z` and `:latest`,
3. **auto-creates the GitHub Release** for `vX.Y.Z` — a header naming the image
   and bundled model, followed by GitHub's auto-generated "What's Changed".

The maintainer no longer writes release notes by hand. `gh release create` fails
if a Release for the tag already exists, so tags are effectively immutable.

## Notes

- The GitHub Release logic lives in `scripts/create-github-release.sh`; its
  behavior is covered by `scripts/test-create-github-release.sh`.
- `pyproject.toml` versions are not the source of truth for the released version
  — the code version is injected from the git tag at build time.
