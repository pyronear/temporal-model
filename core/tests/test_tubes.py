"""Tests for temporal_model.core.tubes."""

import pytest

from temporal_model.core import merge_colocated_tubes as _merge_from_pkg_top
from temporal_model.core.tubes import (
    build_tubes,
    compute_iou,
    interpolate_gaps,
    match_detections,
    merge_colocated_tubes,
    select_longest_tube,
    tube_from_record,
)
from temporal_model.core.types import Detection, FrameDetections, Tube, TubeEntry


def _det(cx: float, cy: float, w: float = 0.1, h: float = 0.1, conf: float = 0.8):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


def _frame(idx: int, detections: list[Detection]):
    return FrameDetections(
        frame_idx=idx,
        frame_id=f"frame_{idx:03d}",
        timestamp=None,
        detections=detections,
    )


# ── compute_iou ──────────────────────────────────────────────────────────


class TestComputeIou:
    def test_identical_boxes(self):
        det = _det(0.5, 0.5, 0.2, 0.2)
        assert abs(compute_iou(det, det) - 1.0) < 1e-9

    def test_no_overlap(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap_iou(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.55, 0.55, 0.2, 0.2)
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = _det(0.5, 0.5, 0.0, 0.0)
        b = _det(0.5, 0.5, 0.1, 0.1)
        assert compute_iou(a, b) == 0.0

    def test_symmetric(self):
        a = _det(0.3, 0.3, 0.2, 0.2)
        b = _det(0.4, 0.4, 0.2, 0.2)
        assert compute_iou(a, b) == compute_iou(b, a)


# ── match_detections ─────────────────────────────────────────────────────


class TestMatchDetections:
    def test_empty_lists(self):
        assert match_detections([], [], 0.1) == []
        assert match_detections([_det(0.5, 0.5)], [], 0.1) == []
        assert match_detections([], [_det(0.5, 0.5)], 0.1) == []

    def test_single_match(self):
        a = _det(0.5, 0.5, 0.2, 0.2)
        b = _det(0.51, 0.51, 0.2, 0.2)
        matches = match_detections([a], [b], 0.1)
        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1] == 0

    def test_below_threshold(self):
        a = _det(0.1, 0.1, 0.1, 0.1)
        b = _det(0.9, 0.9, 0.1, 0.1)
        matches = match_detections([a], [b], 0.1)
        assert len(matches) == 0

    def test_greedy_one_to_one(self):
        a1 = _det(0.3, 0.3, 0.2, 0.2)
        a2 = _det(0.7, 0.7, 0.2, 0.2)
        b1 = _det(0.31, 0.31, 0.2, 0.2)
        b2 = _det(0.71, 0.71, 0.2, 0.2)
        matches = match_detections([a1, a2], [b1, b2], 0.1)
        assert len(matches) == 2
        matched_prev = {m[0] for m in matches}
        matched_curr = {m[1] for m in matches}
        assert matched_prev == {0, 1}
        assert matched_curr == {0, 1}


# ── build_tubes ──────────────────────────────────────────────────────────


class TestBuildTubes:
    def test_empty_sequence(self):
        tubes = build_tubes([])
        assert tubes == []

    def test_no_detections(self):
        frames = [_frame(0, []), _frame(1, []), _frame(2, [])]
        tubes = build_tubes(frames)
        assert tubes == []

    def test_single_detection_single_frame(self):
        frames = [_frame(0, [_det(0.5, 0.5)])]
        tubes = build_tubes(frames)
        assert len(tubes) == 1
        assert len(tubes[0].entries) == 1
        assert tubes[0].entries[0].detection is not None

    def test_consistent_detection_forms_one_tube(self):
        """Same bbox across 5 frames should produce a single tube."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [_frame(i, [det]) for i in range(5)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 1
        assert len(tubes[0].entries) == 5
        assert all(e.detection is not None for e in tubes[0].entries)

    def test_two_separate_detections_form_two_tubes(self):
        """Two non-overlapping detections should form two tubes."""
        d1 = _det(0.2, 0.2, 0.1, 0.1)
        d2 = _det(0.8, 0.8, 0.1, 0.1)
        frames = [_frame(i, [d1, d2]) for i in range(3)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 2

    def test_gap_creates_none_entries(self):
        """Detection missing in middle frame -> gap entry with None."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),  # gap
            _frame(2, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=2)
        assert len(tubes) == 1
        tube = tubes[0]
        # Entries: frame 0 (det), frame 1 (gap), frame 2 (det)
        assert len(tube.entries) == 3
        assert tube.entries[0].detection is not None
        assert tube.entries[1].detection is None
        assert tube.entries[2].detection is not None

    def test_tube_terminated_after_max_misses(self):
        """Tube should be terminated after max_misses consecutive gaps."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),
            _frame(2, []),
            _frame(3, []),  # 3 consecutive misses
            _frame(4, [det]),  # should start a new tube
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=2)
        assert len(tubes) == 2

    def test_max_misses_zero_no_gaps_tolerated(self):
        """With max_misses=0, a single gap terminates the tube."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, [det]),
            _frame(1, []),  # gap
            _frame(2, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1, max_misses=0)
        assert len(tubes) == 2

    def test_tube_start_end_frames(self):
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [_frame(i, [det]) for i in range(5)]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert tubes[0].start_frame == 0
        assert tubes[0].end_frame == 4

    def test_tube_ids_are_unique(self):
        d1 = _det(0.2, 0.2, 0.1, 0.1)
        d2 = _det(0.8, 0.8, 0.1, 0.1)
        frames = [_frame(i, [d1, d2]) for i in range(3)]
        tubes = build_tubes(frames)
        ids = [t.tube_id for t in tubes]
        assert len(ids) == len(set(ids))

    def test_detection_appears_mid_sequence(self):
        """A detection appearing at frame 3 starts a new tube there."""
        det = _det(0.5, 0.5, 0.1, 0.1)
        frames = [
            _frame(0, []),
            _frame(1, []),
            _frame(2, []),
            _frame(3, [det]),
            _frame(4, [det]),
        ]
        tubes = build_tubes(frames, iou_threshold=0.1)
        assert len(tubes) == 1
        assert tubes[0].start_frame == 3
        assert tubes[0].end_frame == 4


# ── select_longest_tube ──────────────────────────────────────────────────


def _make_tube(tube_id: int, length: int, n_gap: int = 0) -> Tube:
    """Build a tube with `length` total entries; last `n_gap` are gaps."""
    n_det = length - n_gap
    entries: list[TubeEntry] = []
    for i in range(n_det):
        entries.append(TubeEntry(frame_idx=i, detection=_det(0.5, 0.5)))
    for i in range(n_gap):
        entries.append(TubeEntry(frame_idx=n_det + i, detection=None, is_gap=True))
    return Tube(
        tube_id=tube_id,
        entries=entries,
        start_frame=0,
        end_frame=length - 1,
    )


class TestSelectLongestTube:
    def test_empty_returns_none(self):
        assert select_longest_tube([]) is None

    def test_picks_longest(self):
        a = _make_tube(0, length=3)
        b = _make_tube(1, length=7)
        c = _make_tube(2, length=5)
        result = select_longest_tube([a, b, c])
        assert result is not None
        assert result.tube_id == 1

    def test_tie_break_by_non_gap_count(self):
        # Both span 5 frames; a has 5 dets, b has only 3 (2 gaps).
        a = _make_tube(0, length=5, n_gap=0)
        b = _make_tube(1, length=5, n_gap=2)
        result = select_longest_tube([a, b])
        assert result is not None
        assert result.tube_id == 0


# ── interpolate_gaps ─────────────────────────────────────────────────────


def _entry(idx: int, det: Detection | None, is_gap: bool = False) -> TubeEntry:
    return TubeEntry(frame_idx=idx, detection=det, is_gap=is_gap)


class TestInterpolateGaps:
    def test_interior_length_one_lerps_midpoint(self):
        before = Detection(0, cx=0.2, cy=0.3, w=0.1, h=0.1, confidence=0.9)
        after = Detection(0, cx=0.4, cy=0.5, w=0.2, h=0.2, confidence=0.7)
        tube = Tube(
            tube_id=0,
            entries=[
                _entry(0, before),
                _entry(1, None, is_gap=True),
                _entry(2, after),
            ],
            start_frame=0,
            end_frame=2,
        )
        out = interpolate_gaps(tube)
        gap = out.entries[1]
        assert gap.is_gap is True
        assert gap.detection is not None
        assert gap.detection.cx == pytest.approx(0.3)
        assert gap.detection.cy == pytest.approx(0.4)
        assert gap.detection.w == pytest.approx(0.15)
        assert gap.detection.h == pytest.approx(0.15)
        assert gap.detection.confidence == pytest.approx(0.0)

    def test_leading_gap_repeats_first_observed(self):
        first = Detection(0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)
        tube = Tube(
            tube_id=0,
            entries=[
                _entry(0, None, is_gap=True),
                _entry(1, first),
            ],
            start_frame=0,
            end_frame=1,
        )
        out = interpolate_gaps(tube)
        g = out.entries[0]
        assert g.is_gap is True
        assert g.detection is not None
        assert g.detection.cx == pytest.approx(0.5)
        assert g.detection.cy == pytest.approx(0.5)
        assert g.detection.confidence == pytest.approx(0.0)

    def test_trailing_gap_repeats_last_observed(self):
        last = Detection(0, cx=0.7, cy=0.2, w=0.05, h=0.05, confidence=0.9)
        tube = Tube(
            tube_id=0,
            entries=[
                _entry(0, last),
                _entry(1, None, is_gap=True),
            ],
            start_frame=0,
            end_frame=1,
        )
        out = interpolate_gaps(tube)
        g = out.entries[1]
        assert g.is_gap is True
        assert g.detection is not None
        assert g.detection.cx == pytest.approx(0.7)
        assert g.detection.confidence == pytest.approx(0.0)

    def test_observed_entries_unchanged(self):
        obs = Detection(0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
        tube = Tube(
            tube_id=0,
            entries=[_entry(0, obs)],
            start_frame=0,
            end_frame=0,
        )
        out = interpolate_gaps(tube)
        assert out.entries[0].is_gap is False
        assert out.entries[0].detection is obs


# ── tube_from_record ─────────────────────────────────────────────────────


class TestTubeFromRecord:
    def test_rebuilds_observed_entry(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": [0.4, 0.5, 0.2, 0.3],
                        "is_gap": False,
                        "confidence": 0.9,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        assert tube.start_frame == 0
        assert tube.end_frame == 0
        assert len(tube.entries) == 1
        e = tube.entries[0]
        assert e.frame_idx == 0
        assert e.is_gap is False
        assert e.detection is not None
        assert e.detection.cx == 0.4
        assert e.detection.cy == 0.5
        assert e.detection.w == 0.2
        assert e.detection.h == 0.3
        assert e.detection.confidence == 0.9

    def test_rebuilds_gap_entry(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": [0.5, 0.5, 0.1, 0.1],
                        "is_gap": True,
                        "confidence": 0.0,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        e = tube.entries[0]
        assert e.is_gap is True
        assert e.detection is not None
        assert e.detection.confidence == 0.0

    def test_rebuilds_missing_bbox_as_none_detection(self):
        record = {
            "tube": {
                "start_frame": 0,
                "end_frame": 0,
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_id": "f0",
                        "bbox": None,
                        "is_gap": True,
                        "confidence": None,
                    }
                ],
            }
        }
        tube = tube_from_record(record)
        e = tube.entries[0]
        assert e.detection is None
        assert e.is_gap is True


# ── merge_colocated_tubes ────────────────────────────────────────────────


def _merge_tube(tid: int, frame_boxes):
    """frame_boxes: list[(frame_idx, (cx, cy, w, h))]."""
    entries = [
        TubeEntry(frame_idx=f, detection=_det(cx, cy, w, h))
        for f, (cx, cy, w, h) in frame_boxes
    ]
    return Tube(
        tube_id=tid,
        entries=entries,
        start_frame=frame_boxes[0][0],
        end_frame=frame_boxes[-1][0],
    )


class TestMergeColocatedTubes:
    KWARGS = dict(merge_iomin=0.3, merge_prox_factor=1.0, merge_max_gap=10)

    def test_empty_input(self):
        assert merge_colocated_tubes([], **self.KWARGS) == []

    def test_single_tube_passthrough(self):
        t = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (1, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([t], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 1)

    def test_merge_contained_overlapping(self):
        """Small box inside big box on shared frames -> one tube."""
        big = _merge_tube(0, [(0, (0.5, 0.5, 0.2, 0.2)), (1, (0.5, 0.5, 0.2, 0.2))])
        small = _merge_tube(1, [(1, (0.5, 0.5, 0.02, 0.02))])
        out = merge_colocated_tubes([big, small], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 1)

    def test_keep_distinct_far_apart(self):
        left = _merge_tube(
            0, [(0, (0.2, 0.5, 0.05, 0.05)), (1, (0.2, 0.5, 0.05, 0.05))]
        )
        right = _merge_tube(
            1, [(0, (0.8, 0.5, 0.05, 0.05)), (1, (0.8, 0.5, 0.05, 0.05))]
        )
        out = merge_colocated_tubes([left, right], **self.KWARGS)
        assert len(out) == 2

    def test_bridge_gap_within_window(self):
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(8, (0.5, 0.5, 0.05, 0.05)), (10, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 10)
        assert any(e.detection is None for e in out[0].entries)

    def test_do_not_bridge_beyond_window(self):
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(16, (0.5, 0.5, 0.05, 0.05)), (18, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b], **self.KWARGS)
        assert len(out) == 2

    def test_transitive_merge(self):
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(3, (0.5, 0.5, 0.05, 0.05))])
        c = _merge_tube(2, [(6, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b, c], **self.KWARGS)
        assert len(out) == 1

    def test_proximity_is_scale_relative(self):
        """Tiny boxes ~1.75 box-sizes apart MUST NOT merge (no teleport)."""
        a = _merge_tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
        near = _merge_tube(
            1, [(2, (0.515, 0.5, 0.02, 0.02)), (3, (0.515, 0.5, 0.02, 0.02))]
        )
        assert len(merge_colocated_tubes([a, near], **self.KWARGS)) == 1
        far = _merge_tube(
            1, [(2, (0.535, 0.5, 0.02, 0.02)), (3, (0.535, 0.5, 0.02, 0.02))]
        )
        assert len(merge_colocated_tubes([a, far], **self.KWARGS)) == 2

    def test_combine_tiebreak_is_order_invariant(self):
        """Equal-area ties resolve to the same box regardless of input order."""
        a = _merge_tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
        b = _merge_tube(1, [(1, (0.51, 0.5, 0.02, 0.02)), (2, (0.51, 0.5, 0.02, 0.02))])

        def box_at_1(tubes):
            [tube] = tubes
            return next(e.detection for e in tube.entries if e.frame_idx == 1)

        assert box_at_1(
            merge_colocated_tubes([a, b], **self.KWARGS)
        ).cx == pytest.approx(0.51)
        assert box_at_1(
            merge_colocated_tubes([b, a], **self.KWARGS)
        ).cx == pytest.approx(0.51)


# ── top-level package exports ────────────────────────────────────────────


def test_merge_colocated_tubes_is_top_level_export():
    """The top-level alias imported at module top must resolve to the same
    function as the one in ``temporal_model.core.tubes``."""
    assert _merge_from_pkg_top is merge_colocated_tubes
