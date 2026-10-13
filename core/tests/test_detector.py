"""Tests for the detector source of truth and typed accessor."""

import pytest
from pydantic import ValidationError

from temporal_model.core.detector import Detector, load_detector


def test_load_detector_returns_expected_identity() -> None:
    det = load_detector()
    assert det.type == "yolo"
    assert det.name == "yolo11s_tenacious-tiger_v8.3.0"
    assert det.source == "hf:pyronear/yolov11s"
    assert det.revision == "v8.3.0"
    assert det.sha256 == (
        "c4b5b846b957cef2465c4f02f657442ec05151b1c3f95cb2e5702c2aa25bc848"
    )


def test_repo_id_strips_hf_prefix() -> None:
    det = load_detector()
    assert det.repo_id == "pyronear/yolov11s"


def test_detector_is_frozen() -> None:
    det = load_detector()
    with pytest.raises(ValidationError):
        det.name = "other"  # type: ignore[misc]


def test_repo_id_rejects_non_hf_source() -> None:
    det = Detector(
        type="yolo", name="x", source="s3://bucket/x", sha256="ab", revision="v1"
    )
    with pytest.raises(ValueError, match="Unsupported detector source"):
        _ = det.repo_id
