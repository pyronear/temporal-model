# Model Versioning — How It Works

How the trained model served by the API is versioned, how its companion detector
is declared and fetched, and how the detector weights are tracked with DVC.

This is the **operational guide**. For the design rationale and the decisions
behind it, see [`docs/specs/2026-06-03-model-versioning-design.md`](specs/2026-06-03-model-versioning-design.md).

---

## 1. The version of a served model

A served model is one self-contained artifact: a packaged `model.zip`. It is
identified by its own **semantic version** (`X.Y.Z`), which is **decoupled from
the repo version**: the API code can change (and be re-released) without
retraining, so the two numbers move independently.

| Where | Value |
|---|---|
| Git tag / Docker image tag | repo version — `vX.Y.Z` / `pyronear/temporal-model-api:X.Y.Z` |
| `/predict` → `version.api` | repo version — `X.Y.Z`, baked into the image from the git tag (`null` on non-release builds) |
| `api/MODEL_VERSION` | **model** version the repo ships — `X.Y.Z` |
| `model.zip` manifest / `/predict` → `version.model` | `model_version: "X.Y.Z"` (matches `api/MODEL_VERSION`) |

The link between the two is the **pin file `api/MODEL_VERSION`**: it names the
model release bundled into the Docker image. Tagging the repo `vX.Y.Z` triggers
the release workflow, which fetches `model.zip` at HF revision
`v$(cat api/MODEL_VERSION)` and tags the image with the repo version.
**Bumping the served model = editing this one file** (plus publishing the new
`model.zip` to HuggingFace first). A repo/code release does **not** imply a
model release. The model's own lineage stays fully recoverable from the
manifest's `provenance` block (below).
At runtime, every `/predict` response reports both identities in one block —
`version: {api, model}` — so a stored result is traceable to the exact
image and model that produced it (see
[`docs/specs/2026-06-11-api-version-in-response-design.md`](specs/2026-06-11-api-version-in-response-design.md)).

## 2. What's inside `model.zip`

`model.zip` is built by `core.package.build_model_package()` and bundles:

```
manifest.yaml          # entry point: version, provenance, file pointers
yolo_weights.pt        # the companion detector (see §3)
classifier.ckpt        # the temporal smoke classifier
config.yaml            # inference config
logistic_calibrator.json   # optional
```

The **manifest** is the shared contract between the writer (`core`/packaging) and
the readers (`api`, `core`). Relevant fields:

```yaml
model_version: "1.4.0"          # omitted on legacy packages → API reports null
provenance:
  train_git_sha: <sha>          # git SHA of the training code
  backbone: vit_small_patch14_dinov2.lvd142m
  detector:                     # copied verbatim from core/detector.yaml (§3)
    type: yolo
    name: yolo11s_nimble-narwhal_v6.0.0
    source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0
    sha256: 0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d
```

All fields are **additive** — packages built before they existed still load, and
the API surfaces a missing `model_version` as `null`.

Artifacts are released on **HuggingFace**, one `model.zip` per version as an HF
revision/tag:

```
huggingface.co/pyronear/temporal-model   (model.zip @ revision v<version>)
```

Publishing a built `model.zip` (which stamps `model_version` and tags the
revision) is an operational step; the API serves whichever package it is baked
with. See the [API release spec](specs/2026-06-03-api-release-design.md).

## 3. The companion detector

The temporal model runs a YOLO detector internally. **Training does not use it**
(`train/build_tubes.py` consumes pre-computed labels), so the detector's identity
cannot be derived from the pipeline — it is **declared** in one place and verified
by hash.

### Source of truth — `core/detector.yaml`

```yaml
detector:
  type: yolo
  name: yolo11s_nimble-narwhal_v6.0.0
  source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0   # downloads best.pt
  sha256: 0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d
```

This is the **only** place the detector is named. `core.detector.load_detector()`
reads and validates it; packaging copies it verbatim into `provenance.detector`.

**Bumping the detector = editing this one file** (a data change, reviewed as a
one-line diff). `type` is generic so the schema survives a detector swap; `name`
follows the pyronear HF convention (`<arch>_<codename>_v<ver>`); `source` is the HF
repo id; `sha256` is the published weights hash and the tamper-evident anchor.

## 4. Fetching a (new) detector

The `fetch_detector` CLI downloads the declared detector from HuggingFace,
**verifies its SHA-256 against `detector.yaml`**, and writes the weights to a path
you choose. It fails loudly on any mismatch.

```bash
cd core
uv run python -m temporal_model.core.fetch_detector \
  --output ../train/data/06_models/detectors/yolo11s_nimble-narwhal_v6.0.0/yolo_weights.pt
```

Output:

```
Fetched yolo11s_nimble-narwhal_v6.0.0 -> .../yolo_weights.pt (sha256 0bf3c7ee… verified)
```

**To switch to a new detector:**

1. Edit `core/detector.yaml` — set `name`, `source`, and the new `sha256` (the
   value published on the detector's HuggingFace model card).
2. Run `fetch_detector` (above) into a new
   `train/data/06_models/detectors/<new-name>/yolo_weights.pt`.
3. Track it with DVC (§5).
4. The next `model.zip` built by packaging will stamp the new identity into
   `provenance.detector` automatically.

## 5. How the detector works with DVC

The detector weights are **DVC-tracked** in the `train` project and pushed to its
existing S3 remote — this gives the repo its own durable, content-addressed copy,
independent of HuggingFace availability, while the small `.dvc` pointer is
committed to git.

The detector is a one-off **import** (declared in `detector.yaml`, fetched on
demand). It is **not a DVC pipeline stage** — `train/dvc.yaml` is detector-free, so
adding or bumping a detector does **not** invalidate the pipeline and you do **not**
need to rerun `dvc repro`.

### Layout

```
train/data/06_models/detectors/<detector-name>/
  yolo_weights.pt        # the 19 MB blob — DVC-tracked, git-ignored
  yolo_weights.pt.dvc    # the pointer — committed to git
```

### Workflow (fetch → track → push → commit)

```bash
# 1. fetch + verify from HuggingFace (writes the blob)
cd core
uv run python -m temporal_model.core.fetch_detector \
  --output ../train/data/06_models/detectors/<name>/yolo_weights.pt

# 2. track it with DVC + push the blob to the S3 remote
cd ../train
export AWS_PROFILE=pyronear   # or rely on direnv / your AWS chain
uv run dvc add  data/06_models/detectors/<name>/yolo_weights.pt
uv run dvc push data/06_models/detectors/<name>/yolo_weights.pt.dvc

# 3. commit the pointer (NOT the blob)
cd ..
git add train/data/06_models/detectors/<name>/yolo_weights.pt.dvc
git commit -m "chore(train): dvc-track detector <name>"
```

`dvc add` records the blob's md5 in the `.dvc` pointer; `dvc push` uploads it to
`s3://pyro-vision-rd/dvc/temporal-model/train/` (the `train` project's `s3remote`).
On another machine, `dvc pull` restores the blob from that pointer.

### `.gitignore` note

The repo ignores all `data/` contents (`**/data/**`) but re-includes the directory
structure, `.gitkeep`, and **`.dvc` pointers**:

```gitignore
**/data/**
!**/data/**/
!**/data/**/.gitkeep
!**/data/**/*.dvc
```

So the 19 MB `yolo_weights.pt` stays out of git while its `yolo_weights.pt.dvc`
pointer is committed normally.

## 6. Verifying an existing artifact

Any `yolo_weights.pt` can be checked against the declared identity by hash:

```bash
sha256sum train/data/06_models/detectors/<name>/yolo_weights.pt
# must equal core/detector.yaml's detector.sha256
```

This is how the currently-served detector was confirmed to be
`yolo11s_nimble-narwhal_v6.0.0` (byte-identical to the HF release).

## 7. What's deferred

Documented here for completeness; not yet implemented (see the spec):

- The **packaging stage** that produces `model.zip` from a trained checkpoint and
  `publish`es it to HuggingFace.
- A maintainer HF write token (local, for `publish`). The `pyronear/temporal-model`
  repo is **public**, so CI `fetch` needs no token.

The **CI release automation** (tag → fetch the pinned `model.zip` from HF → bake
into image → push to Docker Hub) is implemented in `.github/workflows/push.yml`.
