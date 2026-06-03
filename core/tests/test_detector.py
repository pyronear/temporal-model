"""Tests for the detector source of truth and typed accessor."""

import pytest
from pydantic import ValidationError

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
    with pytest.raises(ValidationError):
        det.name = "other"  # type: ignore[misc]


def test_repo_id_rejects_non_hf_source() -> None:
    det = Detector(type="yolo", name="x", source="s3://bucket/x", sha256="ab")
    with pytest.raises(ValueError, match="Unsupported detector source"):
        _ = det.repo_id
