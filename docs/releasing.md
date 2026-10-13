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

### 1. (Only if the model changed) bump the model version

Merge a PR that sets the new version in `api/MODEL_VERSION`:

```bash
echo "X.Y.Z" > api/MODEL_VERSION   # the new model version
```

The model that gets published is the `model.zip` that `train`'s `package`
stage produced at the tagged commit (its `dvc.lock` entry), so merge the
`result_*` PR first. Publishing happens in CI at step 2.

### 2. Push the git tag

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

That triggers `.github/workflows/push.yml`, which:

1. publishes the model if `v<MODEL_VERSION>` is not yet on HuggingFace: pulls
   `train`'s `model.zip` from DVC, exports the torch-free `model_onnx.zip` (the
   export checks torch/ONNX parity and records the source SHA-256), then runs
   `release publish`, which stamps the version into both manifests, uploads the
   zips + model card, and tags `v<MODEL_VERSION>` on the HF repo last. A code-only release
   skips this step,
2. fetches the `model.zip` pinned by `api/MODEL_VERSION` from HuggingFace and
   builds and pushes `pyronear/temporal-model-api:X.Y.Z` and `:latest`,
3. syncs `demo/` to the HuggingFace Space
   [`pyronear/Pyronear-Wildfire-Detection`](https://huggingface.co/spaces/pyronear/Pyronear-Wildfire-Detection)
   and restarts it, so it serves the latest model releases,
4. **auto-creates the GitHub Release** for `vX.Y.Z` — a header naming the image
   and bundled model, followed by GitHub's auto-generated "What's Changed".

The maintainer no longer writes release notes by hand. `gh release create` fails
if a Release for the tag already exists, so tags are effectively immutable.

## Notes

- The CI publish and Space sync need the `HF_TOKEN` secret to have **write**
  access to `pyronear/temporal-model` and the Space. Consumers fetch the ONNX artifact with
  `release fetch --onnx` (or `make fetch-model-onnx`).
- The GitHub Release logic lives in `scripts/create-github-release.sh`; its
  behavior is covered by `scripts/test-create-github-release.sh`.
- `pyproject.toml` versions are not the source of truth for the released version
  — the code version is injected from the git tag at build time.
