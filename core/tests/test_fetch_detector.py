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

    with (
        patch(
            "temporal_model.core.fetch_detector.hf_hub_download",
            return_value=str(src),
        ),
        pytest.raises(ValueError, match="SHA-256 mismatch"),
    ):
        fetch_detector(out, det)
    assert not out.exists()
