"""Integrity guards: detector-weights hash + dvc dep/detector.yaml consistency."""

import hashlib
from pathlib import Path

import pytest
import yaml

from temporal_model.core.detector import Detector, load_detector
from temporal_model.train.package import verify_detector_weights

_TRAIN_DIR = Path(__file__).resolve().parents[1]


def test_verify_detector_weights_passes_on_match(tmp_path: Path) -> None:
    p = tmp_path / "yolo_weights.pt"
    p.write_bytes(b"weights")
    det = Detector(
        type="yolo",
        name="x",
        source="hf:o/x",
        sha256=hashlib.sha256(b"weights").hexdigest(),
    )
    verify_detector_weights(p, det)  # must not raise


def test_verify_detector_weights_raises_on_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "yolo_weights.pt"
    p.write_bytes(b"weights")
    det = Detector(type="yolo", name="x", source="hf:o/x", sha256="00" * 32)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_detector_weights(p, det)


def test_dvc_package_stage_detector_dep_matches_detector_yaml() -> None:
    """Guard the coupling: the dvc.yaml package dep path must track detector.yaml.

    If the detector is bumped in core/detector.yaml but the dvc.yaml dep path is
    not updated, this fails — catching a silently-stale DVC dependency.
    """
    dvc = yaml.safe_load((_TRAIN_DIR / "dvc.yaml").read_text())
    deps = dvc["stages"]["package"]["deps"]
    expected = f"data/06_models/detectors/{load_detector().name}/yolo_weights.pt"
    assert expected in deps, f"{expected!r} not in package stage deps: {deps}"
