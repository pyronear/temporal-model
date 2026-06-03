# Model Versioning (core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every packaged `model.zip` self-describe its version and provenance — a stamped `model_version`, a `provenance` block (train SHA, backbone, hash-verified detector), a single committed detector source of truth, and a reproducible detector-import CLI.

**Architecture:** Add a `core/detector.yaml` source of truth + typed `load_detector()` accessor; a `fetch_detector` CLI that downloads the detector weights from HuggingFace and verifies their SHA-256; and extend `build_model_package()` to write `model_version` + `provenance` (with the detector copied verbatim from `load_detector()`) into the manifest. All changes are additive and backward compatible — existing packages without these fields still load.

**Tech Stack:** Python 3.11, pydantic v2, PyYAML, `huggingface_hub`, pytest, ruff, uv. All work is in the `core` package.

**Scope note:** This plan covers only the in-scope core from the design spec (`docs/specs/2026-06-03-model-versioning-design.md`). The S3 bucket `pyronear-temporal-model` already exists (operational); *uploading* a built `model.zip` to `s3://pyronear-temporal-model/models/<version>/` is an operational step, not code here. The container registry and CI release automation are explicitly deferred.

---

## File structure

| File | Responsibility |
|---|---|
| `core/src/temporal_model/core/detector.yaml` | **Create.** Single source of truth for the bundled detector identity. |
| `core/src/temporal_model/core/detector.py` | **Create.** `Detector` pydantic model + `load_detector()` accessor + constants. |
| `core/src/temporal_model/core/fetch_detector.py` | **Create.** CLI: download detector weights from HF, verify SHA-256, write to an output path. |
| `core/src/temporal_model/core/package.py` | **Modify.** `build_model_package()` gains `model_version` + `train_git_sha` params and writes `model_version` + `provenance` into the manifest. |
| `core/pyproject.toml` | **Modify.** Add `huggingface-hub` as a direct dependency. |
| `core/tests/test_detector.py` | **Create.** Tests for `Detector` + `load_detector()`. |
| `core/tests/test_fetch_detector.py` | **Create.** Tests for `fetch_detector` (download mocked). |
| `core/tests/test_package.py` | **Modify.** Add a `TestProvenance` class for the new manifest fields. |

All test commands run from the `core/` directory (mirrors CI `working-directory`).

---

## Task 1: Detector source of truth + typed accessor

**Files:**
- Create: `core/src/temporal_model/core/detector.yaml`
- Create: `core/src/temporal_model/core/detector.py`
- Test: `core/tests/test_detector.py`

- [ ] **Step 1: Create the detector source-of-truth YAML**

Create `core/src/temporal_model/core/detector.yaml`:

```yaml
# The companion detector bundled into every packaged model.zip.
# Bumping the detector is a data change — edit this file, nothing else.
# Identity is verified by SHA-256 against the published HuggingFace weights.
# See docs/specs/2026-06-03-model-versioning-design.md.
detector:
  type: yolo
  name: yolo11s_nimble-narwhal_v6.0.0
  source: hf:pyronear/yolo11s_nimble-narwhal_v6.0.0   # downloads best.pt
  sha256: 0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d
```

- [ ] **Step 2: Write the failing tests**

Create `core/tests/test_detector.py`:

```python
"""Tests for the detector source of truth and typed accessor."""

import pytest

from temporal_model.core.detector import Detector, load_detector


def test_load_detector_returns_expected_identity() -> None:
    det = load_detector()
    assert det.type == "yolo"
    assert det.name == "yolo11s_nimble-narwhal_v6.0.0"
    assert det.source == "hf:pyronear/yolo11s_nimble-narwhal_v6.0.0"
    assert det.sha256 == (
        "0bf3c7ee9f720c26613c30719fea32f47ed04fc384e443de72414d9f8148ac9d"
    )


def test_repo_id_strips_hf_prefix() -> None:
    det = load_detector()
    assert det.repo_id == "pyronear/yolo11s_nimble-narwhal_v6.0.0"


def test_detector_is_frozen() -> None:
    det = load_detector()
    with pytest.raises(Exception):
        det.name = "other"  # type: ignore[misc]


def test_repo_id_rejects_non_hf_source() -> None:
    det = Detector(type="yolo", name="x", source="s3://bucket/x", sha256="ab")
    with pytest.raises(ValueError, match="Unsupported detector source"):
        _ = det.repo_id
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_model.core.detector'`

- [ ] **Step 4: Implement `detector.py`**

Create `core/src/temporal_model/core/detector.py`:

```python
"""Companion detector: single source of truth + typed accessor.

The bundled YOLO detector identity is declared once in ``detector.yaml`` and
propagated into each packaged model's manifest provenance. It cannot be derived
from training (the training pipeline does not run YOLO), so it is declared here
and verified by SHA-256 against the published HuggingFace weights.

See ``docs/specs/2026-06-03-model-versioning-design.md``.
"""

from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict

DETECTOR_YAML_FILENAME = "detector.yaml"
# The weights file published in the HF detector repo (pyronear convention).
DETECTOR_WEIGHTS_FILENAME = "best.pt"
_HF_PREFIX = "hf:"


class Detector(BaseModel):
    """Identity of the companion detector bundled into ``model.zip``."""

    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    source: str
    sha256: str

    @property
    def repo_id(self) -> str:
        """The HF repo id, e.g. ``pyronear/yolo11s_nimble-narwhal_v6.0.0``."""
        if not self.source.startswith(_HF_PREFIX):
            raise ValueError(
                f"Unsupported detector source: {self.source!r} "
                f"(expected '{_HF_PREFIX}<org>/<repo>')"
            )
        return self.source[len(_HF_PREFIX) :]


def load_detector() -> Detector:
    """Read and validate the detector source of truth (``detector.yaml``)."""
    text = (files("temporal_model.core") / DETECTOR_YAML_FILENAME).read_text()
    data = yaml.safe_load(text)
    return Detector.model_validate(data["detector"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd core && uv run pytest tests/test_detector.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Lint, format, commit**

Run: `cd core && uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

```bash
git add core/src/temporal_model/core/detector.yaml \
        core/src/temporal_model/core/detector.py \
        core/tests/test_detector.py
git commit -m "feat(core): add detector source of truth and typed loader"
```

---

## Task 2: Detector-import CLI with SHA-256 verification

**Files:**
- Modify: `core/pyproject.toml` (add `huggingface-hub`)
- Create: `core/src/temporal_model/core/fetch_detector.py`
- Test: `core/tests/test_fetch_detector.py`

- [ ] **Step 1: Add the `huggingface-hub` dependency**

In `core/pyproject.toml`, add `huggingface-hub` to the `dependencies` list (it is currently present only transitively via `timm`; we import it directly so declare it). The list becomes:

```toml
dependencies = [
    "numpy>=1.26,<2",
    "pydantic>=2.6",
    "torch>=2.2",
    "torchvision>=0.17",
    "timm>=1.0",
    "ultralytics>=8.3",
    "pillow>=10.0",
    "pyyaml>=6.0",
    "huggingface-hub>=0.23",
]
```

- [ ] **Step 2: Lock the new dependency**

Run: `cd core && uv lock`
Expected: `core/uv.lock` updated; exit 0. (`huggingface-hub` already resolves transitively, so this is a no-op resolution change or a small marker update.)

- [ ] **Step 3: Write the failing tests**

Create `core/tests/test_fetch_detector.py`:

```python
"""Tests for the detector-import CLI (download is mocked)."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from temporal_model.core.detector import Detector
from temporal_model.core.fetch_detector import fetch_detector


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_verifies_hash_and_writes_output(tmp_path: Path) -> None:
    weights = b"pretend-yolo-weights"
    src = tmp_path / "best.pt"
    src.write_bytes(weights)
    det = Detector(
        type="yolo",
        name="test-detector",
        source="hf:org/test-detector",
        sha256=_sha256_bytes(weights),
    )
    out = tmp_path / "yolo_weights.pt"

    with patch(
        "temporal_model.core.fetch_detector.hf_hub_download",
        return_value=str(src),
    ) as mock_dl:
        result = fetch_detector(out, det)

    mock_dl.assert_called_once_with(repo_id="org/test-detector", filename="best.pt")
    assert result == out
    assert out.read_bytes() == weights


def test_fetch_raises_on_hash_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "best.pt"
    src.write_bytes(b"actual-bytes")
    det = Detector(
        type="yolo",
        name="test-detector",
        source="hf:org/test-detector",
        sha256=_sha256_bytes(b"DIFFERENT-expected-bytes"),
    )
    out = tmp_path / "yolo_weights.pt"

    with patch(
        "temporal_model.core.fetch_detector.hf_hub_download",
        return_value=str(src),
    ):
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            fetch_detector(out, det)
    assert not out.exists()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_fetch_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_model.core.fetch_detector'`

- [ ] **Step 5: Implement `fetch_detector.py`**

Create `core/src/temporal_model/core/fetch_detector.py`:

```python
"""Download the declared companion detector from HuggingFace and verify it.

Reads the detector identity from ``detector.yaml`` (the single source of truth),
downloads its weights file from the HuggingFace repo, asserts the SHA-256 matches
the declared value, and writes the verified weights to an output path (where the
packaging step expects ``yolo_weights.pt``). Reproducible and tamper-evident.

Usage:
    python -m temporal_model.core.fetch_detector --output yolo_weights.pt
"""

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from .detector import DETECTOR_WEIGHTS_FILENAME, Detector, load_detector


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_detector(output_path: Path, detector: Detector | None = None) -> Path:
    """Download the detector weights, verify the SHA-256, write to ``output_path``.

    Raises:
        ValueError: if the downloaded weights' SHA-256 does not match the
            declared ``detector.sha256``.
    """
    det = detector or load_detector()
    downloaded = Path(
        hf_hub_download(repo_id=det.repo_id, filename=DETECTOR_WEIGHTS_FILENAME)
    )
    actual = _sha256(downloaded)
    if actual != det.sha256:
        raise ValueError(
            f"Detector SHA-256 mismatch for {det.name}: "
            f"expected {det.sha256}, got {actual}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the verified detector weights (e.g. yolo_weights.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    det = load_detector()
    out = fetch_detector(args.output, det)
    print(f"Fetched {det.name} -> {out} (sha256 {det.sha256} verified)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd core && uv run pytest tests/test_fetch_detector.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Lint, format, commit**

Run: `cd core && uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

```bash
git add core/pyproject.toml core/uv.lock \
        core/src/temporal_model/core/fetch_detector.py \
        core/tests/test_fetch_detector.py
git commit -m "feat(core): add fetch_detector CLI with SHA-256 verification"
```

---

## Task 3: Stamp `model_version` + `provenance` into the manifest

**Files:**
- Modify: `core/src/temporal_model/core/package.py` (signature + manifest of `build_model_package`)
- Test: `core/tests/test_package.py` (add `TestProvenance`)

- [ ] **Step 1: Write the failing tests**

First, add the detector import to the **top** of `core/tests/test_package.py`, immediately **before** the existing `from temporal_model.core.logistic_calibrator import LogisticCalibrator` line (ruff isort orders first-party imports alphabetically, so `detector` precedes `logistic_calibrator`; keeping all imports at module top also satisfies `E402`):

```python
from temporal_model.core.detector import load_detector
```

Then append the test class to the end of `core/tests/test_package.py` (reuses the existing `built_archive`, `dummy_yolo_weights`, `dummy_classifier_ckpt` fixtures and `SAMPLE_CONFIG`):

```python
class TestProvenance:
    def test_model_version_recorded_when_provided(
        self,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        dummy_classifier_ckpt: Path,
    ) -> None:
        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="vit_dinov2_finetune",
            output_path=out,
            model_version="1.4.0",
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["model_version"] == "1.4.0"

    def test_model_version_absent_when_not_provided(
        self, built_archive: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert "model_version" not in manifest

    def test_provenance_detector_matches_source_of_truth(
        self, built_archive: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["provenance"]["detector"] == load_detector().model_dump()

    def test_provenance_backbone_from_config(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert (
            manifest["provenance"]["backbone"]
            == SAMPLE_CONFIG["classifier"]["backbone"]
        )

    def test_provenance_train_git_sha_recorded(
        self,
        tmp_path: Path,
        dummy_yolo_weights: Path,
        dummy_classifier_ckpt: Path,
    ) -> None:
        out = tmp_path / "m.zip"
        build_model_package(
            yolo_weights_path=dummy_yolo_weights,
            classifier_ckpt_path=dummy_classifier_ckpt,
            config=SAMPLE_CONFIG,
            variant="vit_dinov2_finetune",
            output_path=out,
            train_git_sha="abc1234",
        )
        with zipfile.ZipFile(out, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["provenance"]["train_git_sha"] == "abc1234"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd core && uv run pytest tests/test_package.py::TestProvenance -v`
Expected: FAIL — `TypeError: build_model_package() got an unexpected keyword argument 'model_version'` (and `KeyError: 'provenance'`).

- [ ] **Step 3: Add the import to `package.py`**

In `core/src/temporal_model/core/package.py`, add to the existing local imports (near the top, immediately **before** `from .logistic_calibrator import LogisticCalibrator` — ruff isort orders these alphabetically, so `.detector` precedes `.logistic_calibrator`):

```python
from .detector import load_detector
```

- [ ] **Step 4: Extend the `build_model_package` signature**

In `core/src/temporal_model/core/package.py`, change the signature to add two keyword-only params (between `output_path` and `calibrator`):

```python
def build_model_package(
    *,
    yolo_weights_path: Path,
    classifier_ckpt_path: Path,
    config: dict[str, Any],
    variant: str,
    output_path: Path,
    model_version: str | None = None,
    train_git_sha: str | None = None,
    calibrator: LogisticCalibrator | None = None,
) -> Path:
```

Add two lines to the docstring's `Args:` section (after the `variant:` entry):

```python
        model_version: Optional released model version, stamped into the
            manifest as ``model_version`` (omitted when ``None``).
        train_git_sha: Optional git SHA of the training code, recorded under
            ``provenance.train_git_sha``.
```

- [ ] **Step 5: Write `model_version` + `provenance` into the manifest**

In `core/src/temporal_model/core/package.py`, locate the manifest construction:

```python
    manifest = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
        "config": CONFIG_FILENAME,
    }
    if calibrator is not None:
        manifest["logistic_calibrator"] = LOGISTIC_CALIBRATOR_FILENAME
```

Insert, immediately after the `if calibrator is not None:` block:

```python
    if model_version is not None:
        manifest["model_version"] = model_version
    manifest["provenance"] = {
        "train_git_sha": train_git_sha,
        "backbone": config.get("classifier", {}).get("backbone"),
        "detector": load_detector().model_dump(),
    }
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `cd core && uv run pytest tests/test_package.py::TestProvenance -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Run the full package test suite (no regressions)**

Run: `cd core && uv run pytest tests/test_package.py -v`
Expected: PASS — all pre-existing tests plus the 5 new ones. (Existing assertions check specific keys/values and the zip namelist, none of which the additive manifest fields change.)

- [ ] **Step 8: Lint, format, commit**

Run: `cd core && uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

```bash
git add core/src/temporal_model/core/package.py core/tests/test_package.py
git commit -m "feat(core): stamp model_version and provenance into model.zip manifest"
```

---

## Task 4: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire core test suite**

Run: `cd core && uv run pytest tests/ -v`
Expected: PASS — all tests across the package green, including `test_detector.py`, `test_fetch_detector.py`, and `test_package.py`.

- [ ] **Step 2: Run lint + format check (mirrors CI)**

Run: `cd core && uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

- [ ] **Step 3: Smoke-check the loader and manifest end to end**

Run:
```bash
cd core && uv run python -c "
from temporal_model.core.detector import load_detector
d = load_detector()
print('detector:', d.name, d.repo_id, d.sha256[:12])
"
```
Expected: prints `detector: yolo11s_nimble-narwhal_v6.0.0 pyronear/yolo11s_nimble-narwhal_v6.0.0 0bf3c7ee9f72`

---

## Self-review notes

- **Spec coverage (in-scope success criteria):**
  1. Manifest carries `model_version` + `provenance.{train_git_sha, backbone, detector}` → Task 3. ✓
  2. `detector.yaml` is the only declaration; `fetch_detector` reproduces & verifies SHA-256 → Tasks 1–2. ✓
  3. `build_model_package()` writes `model_version` + `provenance` → Task 3. ✓
  4. Per-version `model.zip` retrievable from S3 → operational (bucket exists); no code in this plan. Noted in Scope.
  5–7. Image tag invariant / registry push / rollback → deferred (registry & release automation), out of this plan.
- **Backward compatibility:** all manifest additions are additive; `load_model_package()` ignores unknown manifest keys (it reads only `format_version`, file pointers, and the optional `logistic_calibrator`), so older and newer packages both load. `model_version` is omitted when not provided, so the API continues to see `null` for legacy packages.
- **Detector/bytes mismatch in synthetic tests:** `build_model_package` stamps `provenance.detector` from `load_detector()` regardless of the (dummy) bytes bundled in unit tests. That is intentional — provenance declares the *intended* detector identity; `fetch_detector` is what guarantees the real bytes match before a real packaging run.
