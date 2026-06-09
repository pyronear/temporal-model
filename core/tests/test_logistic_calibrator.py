"""Tests for the pure-numpy runtime logistic calibrator module."""

import json
from pathlib import Path

import numpy as np
import pytest

from temporal_model.core.logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
    tube_feature_dict,
)
from temporal_model.core.types import Detection, Tube, TubeEntry


def test_json_roundtrip_preserves_weights(tmp_path: Path) -> None:
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([0.69138, 1.672666, 2.449859, -0.01339]),
        intercept=-6.172852,
        sanity_checks=[],
    )
    out = tmp_path / "calibrator.json"
    cal.to_json(out)
    loaded = LogisticCalibrator.from_json(out)

    assert loaded.features == cal.features
    assert np.allclose(loaded.coefficients, cal.coefficients, atol=1e-12)
    assert loaded.intercept == cal.intercept
    assert loaded.sanity_checks == cal.sanity_checks


def test_json_file_shape_matches_contract(tmp_path: Path) -> None:
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([0.1, 0.2, 0.3, 0.4]),
        intercept=-1.0,
        sanity_checks=[{"features": [1.0, 2.0, 3.0, 4.0], "prob": 0.5}],
    )
    out = tmp_path / "cal.json"
    cal.to_json(out)
    payload = json.loads(out.read_text())

    assert set(payload.keys()) == {
        "features",
        "coefficients",
        "intercept",
        "sanity_checks",
    }
    assert payload["features"] == ["logit", "log_len", "mean_conf", "n_tubes"]
    assert payload["coefficients"] == [0.1, 0.2, 0.3, 0.4]
    assert payload["intercept"] == -1.0
    assert payload["sanity_checks"] == [{"features": [1.0, 2.0, 3.0, 4.0], "prob": 0.5}]


def test_predict_proba_matches_textbook_sigmoid() -> None:
    # Known weights: z = 1*2 + 2*3 + 0 = 8; sigmoid(8) ~= 0.99966
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[],
    )
    prob = cal.predict_proba(np.array([2.0, 3.0]))
    assert abs(prob - 1.0 / (1.0 + np.exp(-8.0))) < 1e-12


def test_predict_proba_batch_matches_single_rows() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([0.5, -0.25]),
        intercept=0.1,
        sanity_checks=[],
    )
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])
    batch = cal.predict_proba_batch(X)
    singles = np.array([cal.predict_proba(row) for row in X])
    assert np.allclose(batch, singles, atol=1e-12)
    assert batch.shape == (3,)


def test_extract_features_matches_analyze_variant_convention() -> None:
    tube = {
        "logit": 2.5,
        "start_frame": 4,
        "end_frame": 12,  # length 9
        "entries": [
            {"confidence": 0.3},
            {"confidence": 0.5},
            {"confidence": None},  # gap entry, must be skipped
            {"confidence": 0.7},
        ],
    }
    feats = extract_features(tube, n_tubes=3)
    # length = 9, log1p(9), mean_conf = (0.3+0.5+0.7)/3 = 0.5
    np.testing.assert_allclose(
        feats,
        np.array([2.5, np.log1p(9), 0.5, 3.0]),
        atol=1e-12,
    )


def test_extract_features_all_none_confidence_defaults_to_zero() -> None:
    tube = {
        "logit": -1.0,
        "start_frame": 0,
        "end_frame": 0,  # length 1
        "entries": [
            {"confidence": None},
            {"confidence": None},
        ],
    }
    feats = extract_features(tube, n_tubes=1)
    np.testing.assert_allclose(
        feats,
        np.array([-1.0, np.log1p(1), 0.0, 1.0]),
        atol=1e-12,
    )


def _reference_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def test_verify_sanity_checks_passes_on_correct_weights() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[
            {"features": [2.0, 3.0], "prob": _reference_sigmoid(8.0)},
            {"features": [0.0, 0.0], "prob": 0.5},
        ],
    )
    cal.verify_sanity_checks()  # no raise


def test_verify_sanity_checks_raises_on_tampered_weights() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[
            {"features": [2.0, 3.0], "prob": _reference_sigmoid(8.0)},
        ],
    )
    tampered = LogisticCalibrator(
        features=cal.features,
        coefficients=cal.coefficients * 2.0,
        intercept=cal.intercept,
        sanity_checks=cal.sanity_checks,
    )
    with pytest.raises(ValueError, match="sanity check"):
        tampered.verify_sanity_checks()


def test_tube_feature_dict_matches_inline_shape():
    det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)
    tube = Tube(
        tube_id=3,
        entries=[
            TubeEntry(frame_idx=0, detection=det),
            TubeEntry(frame_idx=1, detection=None, is_gap=True),
        ],
        start_frame=0,
        end_frame=1,
    )
    d = tube_feature_dict(tube, logit=1.25)
    assert d == {
        "logit": 1.25,
        "start_frame": 0,
        "end_frame": 1,
        "entries": [{"confidence": 0.8}, {"confidence": None}],
    }
    # And it feeds extract_features without error.
    feats = extract_features(d, n_tubes=2)
    assert feats.shape == (4,)
