"""Unit tests for the pure stable-crop-window helper."""

import pytest

from temporal_model.core.stabilize import union_window


def test_union_of_two_boxes_is_axis_independent():
    # A: x[0.15,0.25] y[0.45,0.55]; B: x[0.35,0.45] y[0.45,0.55]
    # union: x[0.15,0.45] (w=0.3, cx=0.3); y[0.45,0.55] (h=0.1, cy=0.5)
    boxes = [(0.2, 0.5, 0.1, 0.1), (0.4, 0.5, 0.1, 0.1)]
    assert union_window(boxes) == pytest.approx((0.3, 0.5, 0.3, 0.1))


def test_single_box_returns_itself():
    assert union_window([(0.5, 0.5, 0.2, 0.2)]) == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_empty_raises():
    with pytest.raises(ValueError):
        union_window([])
