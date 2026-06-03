# Trained-Model Versioning & Release — Design

**Date:** 2026-06-03
**Status:** Draft
**Scope:** How a trained temporal smoke model (`model.zip`) is versioned, stored,
released, and deployed to the API. Covers the version identity, manifest
provenance (including detector tracking), artifact storage, the Docker-image
deployable, and the release/rollback flow. The step that *produces* `model.zip`
from a trained checkpoint (packaging) is a separate future stage and is out of
scope except for the manifest contract it must satisfy.

## Goal

Replace today's "the `model.zip` just shows up, gitignored and untracked" with a
versioning and release story that is:

1. **Identifiable** — every served model has one version string, used identically
   across the git tag, the Docker image tag, and `manifest.model_version`.
2. **Traceable** — the manifest self-describes how the model was built (train code
   SHA, backbone, and the exact companion detector, verified by hash).
3. **Stored** — every versioned `model.zip` has a durable home on HuggingFace
   (`pyronear/temporal-model`, one revision/tag per version), retrievable by version.
4. **Deployable & reversible** — a release produces a self-contained Docker image
   tagged with the version; deploy = pin a tag, rollback = redeploy a prior tag.

## Background

Established facts about the current repo (see
`docs/specs/2026-06-02-api-service-design.md` and `core/package.py`):

- The API loads one `model.zip` at startup from `MODEL_PATH` and reads
  `manifest.get("model_version")` → surfaces it as `model.version` (today: `null`).
- `model.zip` bundles `manifest.yaml`, `yolo_weights.pt`, `classifier.ckpt`,
  `config.yaml`, optional `logistic_calibrator.json`. The detector runs *inside*
  the model.
- `core/package.py::build_model_package()` writes a manifest with
  `format_version`, `variant`, and file pointers only — **no `model_version`, no
  provenance, no detector identity** beyond the bundled filename.
- The API spec **explicitly deferred** the versioning scheme to "the training
  spec." This document is that decision.

**The training pipeline does not use YOLO.** `train/build_tubes.py` consumes
pre-computed label files ("No YOLO inference is performed — the labels carry
everything we need"). The companion detector enters only at *packaging* time, as a
`yolo_weights.pt` supplied by the caller. Its identity is therefore not derivable
from training and must be **declared** in the repo and propagated into the manifest.

**The currently-served detector is `pyronear/yolo11s_nimble-narwhal_v6.0.0`**
(file `best.pt`), confirmed by downloading the HF release and comparing hashes —
**byte-identical** to the bundled `yolo_weights.pt`:

| | SHA-256 | MD5 | size |
|---|---|---|---|
| HF `nimble-narwhal_v6.0.0/best.pt` | `0bf3c7ee…ac9d` | `fea4ddc8…0837` | 19,225,626 |
| bundled `yolo_weights.pt` | `0bf3c7ee…ac9d` | `fea4ddc8…0837` | 19,225,626 |

The HF codename/version is *not* stored in the checkpoint (only arch `yolo11s`,
ultralytics 8.4.21, and a `pyro-train` commit `65ba4f9`), so the only reliable
bytes→release mapping is the published SHA-256. This motivates hash-verified
detector tracking below.

## Decisions

| Decision | Choice |
|---|---|
| Version string | One semver `X.Y.Z`, identical across git tag (`vX.Y.Z`), Docker image tag, and `manifest.model_version`. Bumps on **either** code or model change — the deployable is one product. |
| Provenance home | A `provenance` block in the manifest, stamped at packaging time. |
| Provenance fields | `train_git_sha`, `backbone`, `detector`. (No `dvc_exp_id`, no `metrics` — those live in the GH Release notes if wanted.) |
| Detector source of truth | `core/detector.yaml` — `{type, name, source, sha256}`, loaded via a typed `load_detector()`. Single mapping, not a registry (YAGNI). |
| Detector reproducibility | An import script downloads the detector weights from HF and **asserts the SHA-256** matches `detector.yaml`. |
| Artifact bytes | **HuggingFace** repo `pyronear/temporal-model`, one `model.zip` at HF revision/tag `v<version>` per release. No S3. (See the [API release spec](2026-06-03-api-release-design.md).) |
| Deployable | Docker image with weights **baked in** at build, tagged **`pyronear/temporal-model-api:<version>`** on **Docker Hub**. Self-contained; no runtime model fetch. |
| Release index | Optional GitHub Release per `vX.Y.Z` (notes + rendered provenance + link to the HF revision). Not required to ship images. |
| Trigger | Push git tag `vX.Y.Z` → CI fetches from HF, bakes, pushes the image. |
| Deploy / rollback | Pin the image tag; rollback = redeploy the prior tag. No runtime model selection. |
| Packaging stage | Out of scope (sibling [packaging spec](2026-06-03-packaging-pipeline-design.md)). Precondition: a `model.zip` is **published** to the HF repo at `v<version>`. |

## Version identity

A single semantic version `X.Y.Z` names the **deployable as one product**. It
appears, byte-for-byte identical, in three places:

- the **git tag**: `vX.Y.Z`
- the **Docker image tag**: `<registry>/temporal-api:X.Y.Z` (registry TBD — follow-up)
- the **manifest**: `model_version: "X.Y.Z"`

It bumps on *either* an API code change or a model change — there is intentionally
no separate "model version" vs "code version" axis. The model's own lineage is
still fully recoverable from the `provenance` block; the human-facing handle is one
number.

CI enforces the invariant: the release build **asserts
`manifest.model_version == <git tag without the leading v>`** and fails otherwise,
so the three sources can never silently drift.

## Manifest contract

The manifest gains two additive fields (existing packages without them remain
loadable; the API already treats a missing `model_version` as `null`):

```yaml
# manifest.yaml (additions)
model_version: "1.4.0"
provenance:
  train_git_sha: "<sha of the temporal-model training code>"
  backbone: "vit_small_patch14_dinov2.lvd142m"
  detector:
    type: yolo
    name: yolo11s_nimble-narwhal_v6.0.0
    source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0
    sha256: 0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d
```

`build_model_package()` (in `core/package.py`) is extended to accept and write
`model_version` and `provenance`. The `detector` sub-block is copied verbatim from
`load_detector()` (below), so the manifest's detector identity is exactly the
verified source of truth. `backbone` comes from the bundled classifier config;
`train_git_sha` is supplied by the packaging step.

This keeps the manifest the **single shared contract** between the writer
(packaging) and the readers (`api`, `core`), consistent with the API spec's
decision that the manifest schema lives in `core`.

## Detector tracking

The companion detector cannot be derived from training, so it is **declared once**
and **propagated**.

### Source of truth — `core/detector.yaml`

```yaml
# core/src/temporal_model/core/detector.yaml
# The companion detector bundled into every packaged model.zip.
# Bumping the detector is a data change — edit this file, nothing else.
detector:
  type: yolo
  name: yolo11s_nimble-narwhal_v6.0.0
  source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0   # downloads best.pt
  sha256: 0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d
```

- **`type`** is generic so the schema survives a detector swap (e.g. `rt-detr`).
- **`name`** is the pyronear HF naming convention, verbatim
  (`<arch>_<codename>_v<ver>`).
- **`source`** is the HF repo id; `hf:<org>/<repo>` resolves to
  `https://huggingface.co/<org>/<repo>`, file `best.pt`.
- **`sha256`** is the published HF weights hash — the tamper-evident fingerprint.

It is a **single mapping**, not a `detectors:` registry. Only one detector ships
today; widening to a keyed registry later is non-breaking.

### Typed loader — `core/detector.py`

```python
class Detector(BaseModel):
    type: str
    name: str
    source: str
    sha256: str

def load_detector() -> Detector: ...   # reads detector.yaml, validates
```

Both consumers go through `load_detector()`:

- **Import script** (`python -m temporal_model.core.fetch_detector`): resolves
  `source` → downloads `best.pt` from HF via `huggingface_hub` →
  **asserts the file's SHA-256 == `detector.sha256`**, failing loudly on mismatch →
  writes `yolo_weights.pt` where packaging expects it. Reproducible and
  tamper-evident: anyone can re-fetch the exact detector from the repo alone.
- **Packaging** stamps `provenance.detector = load_detector().model_dump()` into
  the manifest.

## Storage & distribution

Two roles, tied together by the one version string:

- **HuggingFace = bytes.** The packaged `model.zip` is released on a HuggingFace
  model repo **`pyronear/temporal-model`** — symmetric with the detector (also on
  HF). One `model.zip`, one HF git **revision/tag `v<version>`** per release; `fetch`
  pins the exact bytes with `revision="v<version>"`. No S3 is used for model
  release. (The `pyro-vision-rd` S3 remote backs DVC/pipeline data only, unrelated.)
- **Image = deployable.** CI bakes that exact `model.zip` into the API image
  (Docker Hub `pyronear/temporal-model-api:<version>`). The running container is
  self-contained — **no model fetch at runtime**. Only CI fetches from HF, at build
  time.

The detailed publish/fetch mechanics, auth, and workflow live in the
[API release spec](2026-06-03-api-release-design.md); the **packaging pipeline**
that produces `model.zip` lives in the
[packaging spec](2026-06-03-packaging-pipeline-design.md).

## Release flow (target shape)

A release is one human action — pushing a tag — followed by CI:

```
(precondition) model.zip published to HF pyronear/temporal-model @ tag vX.Y.Z

push git tag vX.Y.Z
  └─► CI:
        1. fetch model.zip from HF @ revision vX.Y.Z     (hf_hub_download)
        2. assert manifest.model_version == X.Y.Z        (fail on mismatch)
        3. build API image with model.zip baked in
        4. push  pyronear/temporal-model-api:X.Y.Z
```

**Deploy:** pin the image tag (`temporal-api:X.Y.Z`).
**Rollback:** redeploy the prior tag. There is no runtime model selection — the
image *is* the version.

## Components / changes

| Unit | Change |
|---|---|
| `core/detector.yaml` | **New.** Single source of truth for the bundled detector. |
| `core/detector.py` | **New.** `Detector` model + `load_detector()`. |
| `core/fetch_detector.py` | **New.** CLI: download detector weights from HF, verify SHA-256, write `yolo_weights.pt`. |
| `core/package.py` | **Extend.** `build_model_package()` accepts/writes `model_version` and `provenance` (with `detector` from `load_detector()`). |
| CI release workflow | Tag-triggered: fetch `model.zip` from HF → assert version → bake into image → push to Docker Hub. See the [API release spec](2026-06-03-api-release-design.md). |
| Container registry | **Docker Hub** `pyronear/temporal-model-api` (decided). |
| Artifact storage | **HuggingFace** `pyronear/temporal-model` (revision per version). The earlier S3 bucket `pyronear-temporal-model` is **unused and can be deleted**. |
| `api` | **No change.** Already reads `manifest.model_version`; can later surface `provenance` via `?verbose=true` or a future `GET /model` (out of scope here). |

## Out of scope (this spec)

- The **packaging stage** that produces `model.zip` from a trained checkpoint and
  the `publish` step that uploads it to HuggingFace. Precondition only: a `model.zip`
  is published to `pyronear/temporal-model` at revision `v<version>`.
- Recording the YOLO model that generated the **training FP labels** (a dataset
  provenance concern, upstream of this repo).
- Surfacing `provenance` through the API response (API spec territory).
- A detector **registry** (multiple coexisting detectors) — single mapping until a
  second detector exists.
- Runtime model selection / hot-swap — incompatible with baked-in images by design.
- The detailed release/publish mechanics — now specified in the
  [API release spec](2026-06-03-api-release-design.md).

## Success criteria

**In scope (this spec):**

1. Every released `model.zip` manifest carries `model_version` and
   `provenance.{train_git_sha, backbone, detector}`.
2. `core/detector.yaml` is the only place the detector identity is declared;
   `fetch_detector` reproduces `yolo_weights.pt` from HF and verifies its SHA-256
   against that file.
3. `core/package.py::build_model_package()` writes `model_version` and
   `provenance` (with `detector` from `load_detector()`) into the manifest.
4. Each version's `model.zip` is retrievable from HuggingFace
   `pyronear/temporal-model` at revision `v<version>`.

**Target (release automation — see the API release spec):**

5. One semver appears identically in the git tag, the image tag, and
   `manifest.model_version`; CI fails a release on any mismatch.
6. A release produces `pyronear/temporal-model-api:<version>` with weights baked
   in, runnable with no model fetch at runtime.
7. Rollback is "redeploy the previous image tag," with no other change.
