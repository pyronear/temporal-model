"""Tests for decision-threshold calibration."""

import numpy as np
import pytest

from temporal_model.train.calibration import calibrate_threshold


def test_threshold_meets_target_recall() -> None:
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 1, 1, 1])
    # 3 positives; target_recall 1.0 → keep all → threshold = smallest pos prob.
    assert calibrate_threshold(probs, labels, target_recall=1.0) == 0.4


def test_threshold_drops_lowest_positive_at_partial_recall() -> None:
    probs = np.array([0.2, 0.5, 0.7, 0.95])
    labels = np.array([0, 1, 1, 1])
    # n_pos=3, target 0.66 → floor(3*0.34)=1 drop → threshold = 2nd-lowest pos = 0.7
    assert calibrate_threshold(probs, labels, target_recall=0.66) == 0.7


def test_no_positives_raises() -> None:
    with pytest.raises(ValueError, match="no positives"):
        calibrate_threshold(np.array([0.1, 0.2]), np.array([0, 0]), target_recall=0.9)


def test_bad_target_recall_raises() -> None:
    with pytest.raises(ValueError, match="target_recall"):
        calibrate_threshold(np.array([0.5]), np.array([1]), target_recall=1.5)


def test_mis_shaped_raises() -> None:
    with pytest.raises(ValueError, match="equal-length 1D"):
        calibrate_threshold(np.array([[0.5]]), np.array([[1]]), target_recall=0.9)
