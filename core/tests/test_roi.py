"""Tests for ROI tube filtering (spec: 2026-06-10-api-roi-design.md)."""

from unittest.mock import MagicMock

import pytest

# Reuse the shared fixtures/helpers from the edge-case suite.
from test_model_edge_cases import (  # type: ignore[import-not-found]
    TEST_CONFIG,
    _fake_yolo_factory,
    red_frames,
    tiny_classifier,
)

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.tubes import tube_intersects_roi
from temporal_model.core.types import Detection, Tube, TubeEntry

__all__ = ["red_frames", "tiny_classifier"]  # keep fixtures importable


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


def _two_cluster_model(red_frames, tiny_classifier):
    """Model whose fake YOLO emits two spatial clusters -> two tubes."""
    per_frame = [
        [(0.2, 0.5, 0.1, 0.1, 0.9), (0.7, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames
    ]
    return BboxTubeTemporalModel(
        yolo_model=_fake_yolo_factory(per_frame),
        classifier=tiny_classifier,
        config=TEST_CONFIG,
    )


class TestPredictWithRoi:
    def test_roi_keeps_only_intersecting_tube(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        out = model.predict(frames=red_frames, roi=(0.6, 0.4, 0.8, 0.6))

        kept = out.details["tubes"]["kept"]
        assert len(kept) == 1
        assert out.details["tubes"]["num_outside_roi"] == 1
        # The surviving tube is the x~0.7 cluster.
        cxs = [e["bbox"][0] for e in kept[0]["entries"] if e["bbox"]]
        assert all(abs(cx - 0.7) < 0.05 for cx in cxs)
        # num_candidates keeps its pre-ROI meaning.
        assert out.details["tubes"]["num_candidates"] == 2

    def test_roi_excluding_everything_is_negative(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        out = model.predict(frames=red_frames, roi=(0.45, 0.05, 0.55, 0.15))

        assert out.is_positive is False
        assert out.details["tubes"]["kept"] == []
        assert out.details["tubes"]["num_outside_roi"] == 2

    def test_roi_none_matches_baseline(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        baseline = model.predict(frames=red_frames)
        out = model.predict(frames=red_frames, roi=None)

        assert out == baseline
        assert baseline.details["tubes"]["num_outside_roi"] == 0

    def test_whole_frame_roi_matches_baseline(self, red_frames, tiny_classifier):
        model = _two_cluster_model(red_frames, tiny_classifier)
        baseline = model.predict(frames=red_frames)
        out = model.predict(frames=red_frames, roi=(0.0, 0.0, 1.0, 1.0))

        assert out == baseline

    def test_roi_with_compute_trigger_searches_only_kept_tubes(
        self, red_frames, tiny_classifier
    ):
        # Guards the filter-before-trigger-search ordering: the trigger must
        # never point at a tube the ROI dropped.
        model = _two_cluster_model(red_frames, tiny_classifier)
        out = model.predict(
            frames=red_frames, roi=(0.6, 0.4, 0.8, 0.6), compute_trigger=True
        )

        kept = out.details["tubes"]["kept"]
        assert len(kept) == 1
        assert out.details["tubes"]["num_outside_roi"] == 1
        trigger_tube_id = out.details["decision"]["trigger_tube_id"]
        if out.is_positive:
            assert trigger_tube_id == kept[0]["tube_id"]
            assert out.trigger_frame_index is not None
        else:
            assert trigger_tube_id is None
            assert out.trigger_frame_index is None


class TestRoiValidation:
    @pytest.mark.parametrize(
        "roi",
        [
            (-0.1, 0.0, 1.0, 1.0),  # out of range low
            (0.0, 0.0, 1.0, 1.1),  # out of range high
            (0.5, 0.2, 0.4, 0.8),  # x_min >= x_max
            (0.2, 0.8, 0.4, 0.8),  # y_min >= y_max (zero height)
            (0.2, 0.8),  # wrong arity
        ],
    )
    def test_invalid_roi_raises(self, roi, tiny_classifier):
        model = BboxTubeTemporalModel(
            yolo_model=MagicMock(),
            classifier=tiny_classifier,
            config=TEST_CONFIG,
        )
        with pytest.raises(ValueError, match="roi"):
            model.predict(frames=[], roi=roi)
