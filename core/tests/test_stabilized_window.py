"""Unit tests for tube_stabilized_window (the persisted stabilized crop window)."""

from dataclasses import dataclass

from temporal_model.core.stabilize import tube_stabilized_window


@dataclass
class _Det:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class _Entry:
    detection: _Det | None
    is_gap: bool = False


def test_window_is_union_of_observed_boxes():
    # Two boxes; the union/enclosing box centers between them and spans both.
    entries = [
        _Entry(_Det(0.2, 0.2, 0.1, 0.1)),
        _Entry(_Det(0.4, 0.4, 0.1, 0.1)),
    ]
    cx, cy, w, h = tube_stabilized_window(entries)
    # x spans 0.15..0.45 -> center 0.30, width 0.30; same for y.
    assert round(cx, 4) == 0.3
    assert round(cy, 4) == 0.3
    assert round(w, 4) == 0.3
    assert round(h, 4) == 0.3


def test_window_ignores_gap_only_entries_without_detection():
    entries = [
        _Entry(None, is_gap=True),
        _Entry(_Det(0.5, 0.5, 0.2, 0.2)),
    ]
    cx, cy, w, h = tube_stabilized_window(entries)
    assert (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)) == (0.5, 0.5, 0.2, 0.2)


def test_window_none_when_no_usable_detection():
    entries = [_Entry(None, is_gap=True), _Entry(None, is_gap=True)]
    assert tube_stabilized_window(entries) is None
