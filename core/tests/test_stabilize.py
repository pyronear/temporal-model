"""Unit tests for the pure stable-crop-window helpers."""

import pytest

from temporal_model.core.stabilize import tube_window, union_window


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


def test_tube_window_unions_non_gap_boxes():
    boxes = [((0.3, 0.5, 0.1, 0.1), False), ((0.7, 0.5, 0.1, 0.1), False)]
    assert tube_window(boxes) == pytest.approx((0.5, 0.5, 0.5, 0.1))


def test_tube_window_ignores_gap_boxes_when_observed_present():
    # The far-away gap box must NOT widen the window.
    boxes = [
        ((0.3, 0.5, 0.1, 0.1), False),
        ((0.9, 0.5, 0.2, 0.2), True),
        ((0.5, 0.5, 0.1, 0.1), False),
    ]
    # observed x[0.25,0.55] -> w=0.3 cx=0.4; y[0.45,0.55] -> h=0.1 cy=0.5
    assert tube_window(boxes) == pytest.approx((0.4, 0.5, 0.3, 0.1))


def test_tube_window_falls_back_to_all_when_only_gaps():
    boxes = [((0.3, 0.5, 0.1, 0.1), True), ((0.7, 0.5, 0.1, 0.1), True)]
    assert tube_window(boxes) == pytest.approx((0.5, 0.5, 0.5, 0.1))


def test_tube_window_ignores_none_boxes():
    boxes = [
        ((0.3, 0.5, 0.1, 0.1), False),
        (None, False),
        ((0.5, 0.5, 0.1, 0.1), False),
    ]
    assert tube_window(boxes) == pytest.approx((0.4, 0.5, 0.3, 0.1))


def test_tube_window_empty_raises():
    with pytest.raises(ValueError):
        tube_window([])


def test_tube_window_all_none_raises():
    with pytest.raises(ValueError):
        tube_window([(None, False), (None, True)])
