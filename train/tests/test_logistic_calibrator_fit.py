"""Tests for the package-time logistic calibrator fitter."""

import pytest

from temporal_model.train.logistic_calibrator_fit import fit


def _record(label: str, logit: float) -> dict:
    """A minimal kept-tube record extract_features can consume."""
    return {
        "label": label,
        "kept_tubes": [
            {
                "logit": logit,
                "start_frame": 0,
                "end_frame": 4,
                "entries": [{"confidence": 0.8}, {"confidence": 0.9}],
            }
        ],
    }


def _records() -> list[dict]:
    smoke = [_record("smoke", logit) for logit in (2.0, 3.0, 4.0, 1.5, 2.5)]
    fp = [_record("fp", logit) for logit in (-2.0, -1.0, -3.0, -0.5, -1.5)]
    return smoke + fp


def test_fit_returns_calibrator_passing_sanity_checks() -> None:
    cal = fit(_records())
    assert len(cal.coefficients) == 4
    assert cal.sanity_checks  # non-empty
    cal.verify_sanity_checks()  # must not raise


def test_fit_single_class_raises() -> None:
    with pytest.raises(ValueError):
        fit([_record("smoke", 2.0), _record("smoke", 3.0)])
