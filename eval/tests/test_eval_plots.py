"""Smoke tests for shared plot helpers."""

import numpy as np

from temporal_model.eval.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def test_plot_confusion_matrix_writes_nonempty_png(tmp_path):
    matrix = np.array([[10, 2], [3, 15]], dtype=float)
    out = tmp_path / "cm.png"

    plot_confusion_matrix(matrix, out, title="smoke test", normalized=False)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix_normalized_writes_nonempty_png(tmp_path):
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    out = tmp_path / "cm_norm.png"

    plot_confusion_matrix(matrix, out, title="smoke test (norm)", normalized=True)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pr_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "pr.png"

    plot_pr_curve(y_true, scores, out, title="PR")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "roc.png"

    plot_roc_curve(y_true, scores, out, title="ROC")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pr_curve_handles_no_positives(tmp_path):
    """Degenerate y_true (all zeros) must still produce a readable PNG."""
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.4, 0.3, 0.2])
    out = tmp_path / "pr_no_pos.png"

    plot_pr_curve(y_true, scores, out, title="PR")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curve_handles_single_class_all_negative(tmp_path):
    """Degenerate y_true (all zeros) must still produce a readable PNG."""
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.4, 0.3, 0.2])
    out = tmp_path / "roc_all_neg.png"

    plot_roc_curve(y_true, scores, out, title="ROC")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curve_handles_single_class_all_positive(tmp_path):
    """Degenerate y_true (all ones) must still produce a readable PNG."""
    y_true = np.array([1, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.3, 0.2])
    out = tmp_path / "roc_all_pos.png"

    plot_roc_curve(y_true, scores, out, title="ROC")

    assert out.exists()
    assert out.stat().st_size > 0
