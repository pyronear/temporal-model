import pytest

from temporal_model.monitor.geometry import tube_stabilized_window, union_window


def entry(bbox, is_gap=False):
    return {"frame_idx": 0, "bbox": bbox, "is_gap": is_gap, "confidence": 0.5}


def test_union_window_single_box_is_identity():
    assert union_window([(0.5, 0.5, 0.2, 0.1)]) == pytest.approx((0.5, 0.5, 0.2, 0.1))


def test_union_window_encloses():
    # box A spans x [0.1, 0.3], box B spans x [0.4, 0.6] -> union x [0.1, 0.6]
    a = (0.2, 0.2, 0.2, 0.2)
    b = (0.5, 0.5, 0.2, 0.2)
    assert union_window([a, b]) == pytest.approx((0.35, 0.35, 0.5, 0.5))


def test_union_window_empty_raises():
    with pytest.raises(ValueError):
        union_window([])


def test_stabilized_window_ignores_gap_boxes():
    observed = (0.2, 0.2, 0.2, 0.2)
    gap = (0.8, 0.8, 0.1, 0.1)  # interpolated — must not widen the window
    win = tube_stabilized_window([entry(list(observed)), entry(list(gap), is_gap=True)])
    assert win == pytest.approx(observed)


def test_stabilized_window_falls_back_to_gap_boxes():
    gap = (0.8, 0.8, 0.1, 0.1)
    win = tube_stabilized_window([entry(None), entry(list(gap), is_gap=True)])
    assert win == pytest.approx(gap)


def test_stabilized_window_none_without_any_box():
    assert tube_stabilized_window([entry(None), entry(None, is_gap=True)]) is None
