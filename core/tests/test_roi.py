"""Tests for ROI tube filtering (spec: 2026-06-10-api-roi-design.md)."""

from temporal_model.core.tubes import tube_intersects_roi
from temporal_model.core.types import Detection, Tube, TubeEntry


def _det(cx: float, cy: float, w: float = 0.1, h: float = 0.1, conf: float = 0.8):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


def _tube(entries: list[TubeEntry]) -> Tube:
    return Tube(tube_id=0, entries=entries, start_frame=0, end_frame=len(entries))


class TestTubeIntersectsRoi:
    def test_overlapping_detection_keeps_tube(self):
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.55, 0.45, 0.9, 0.55)) is True

    def test_touching_edge_counts_as_overlap(self):
        # Detection box right edge at x=0.6 exactly touches roi x_min=0.6.
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.6, 0.4, 0.9, 0.6)) is True

    def test_fully_outside_drops_tube(self):
        tube = _tube([TubeEntry(frame_idx=0, detection=_det(0.5, 0.5, 0.2, 0.2))])
        assert tube_intersects_roi(tube, (0.7, 0.7, 0.9, 0.9)) is False

    def test_gap_entries_do_not_count(self):
        # The only entry overlapping the ROI is a gap (synthetic, lerped bbox);
        # the sole real detection is outside. Tube must be dropped.
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=_det(0.5, 0.5), is_gap=True),
                TubeEntry(frame_idx=1, detection=_det(0.1, 0.1)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is False

    def test_pre_interpolation_gap_without_detection_is_ignored(self):
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=None, is_gap=True),
                TubeEntry(frame_idx=1, detection=_det(0.5, 0.5)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is True

    def test_any_single_real_entry_inside_keeps_tube(self):
        # First entries outside, last one drifted into the ROI.
        tube = _tube(
            [
                TubeEntry(frame_idx=0, detection=_det(0.1, 0.1)),
                TubeEntry(frame_idx=1, detection=_det(0.2, 0.2)),
                TubeEntry(frame_idx=2, detection=_det(0.5, 0.5)),
            ]
        )
        assert tube_intersects_roi(tube, (0.45, 0.45, 0.55, 0.55)) is True
