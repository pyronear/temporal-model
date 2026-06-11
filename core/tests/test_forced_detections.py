"""Tests for make_forced_detections (caller-supplied bbox, no YOLO)."""

from unittest.mock import MagicMock

import pytest

# Reuse the shared fixtures/helpers from the edge-case suite.
from test_model_edge_cases import (  # type: ignore[import-not-found]
    TEST_CONFIG,
    red_frames,
    tiny_classifier,
)

from temporal_model.core.inference import make_forced_detections
from temporal_model.core.model import BboxTubeTemporalModel

__all__ = ["red_frames", "tiny_classifier"]  # keep fixtures importable

BBOX = (0.4, 0.4, 0.6, 0.6)


def test_one_single_detection_entry_per_frame(red_frames):
    dets = make_forced_detections(red_frames, bbox_xyxyn=BBOX)

    assert set(dets) == {f.frame_id for f in red_frames}
    for idx, f in enumerate(red_frames):
        fd = dets[f.frame_id]
        assert fd.frame_idx == idx
        assert fd.frame_id == f.frame_id
        assert len(fd.detections) == 1


def test_xyxyn_converted_to_center_format(red_frames):
    dets = make_forced_detections(
        red_frames, bbox_xyxyn=(0.1, 0.2, 0.5, 0.8), confidence=0.7
    )

    d = dets[red_frames[0].frame_id].detections[0]
    assert d.cx == pytest.approx(0.3)
    assert d.cy == pytest.approx(0.5)
    assert d.w == pytest.approx(0.4)
    assert d.h == pytest.approx(0.6)
    assert d.confidence == 0.7


def test_confidence_defaults_to_one(red_frames):
    dets = make_forced_detections(red_frames, bbox_xyxyn=BBOX)
    assert dets[red_frames[0].frame_id].detections[0].confidence == 1.0


@pytest.mark.parametrize(
    "bbox",
    [
        (0.6, 0.4, 0.4, 0.6),  # x_min >= x_max
        (0.4, 0.6, 0.6, 0.4),  # y_min >= y_max
        (-0.1, 0.0, 0.5, 0.5),  # out of [0, 1]
    ],
)
def test_invalid_bbox_raises(red_frames, bbox):
    with pytest.raises(ValueError, match="invalid bbox"):
        make_forced_detections(red_frames, bbox_xyxyn=bbox)


def test_predict_with_forced_detections_skips_yolo_and_builds_one_tube(
    red_frames, tiny_classifier
):
    yolo = MagicMock()
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    forced = make_forced_detections(red_frames, bbox_xyxyn=BBOX)
    out = model.predict(frames=red_frames, frame_detections=forced)

    yolo.predict.assert_not_called()
    kept = out.details["tubes"]["kept"]
    assert len(kept) == 1
    assert kept[0]["start_frame"] == 0
    assert kept[0]["end_frame"] == len(red_frames) - 1
    assert all(not e["is_gap"] for e in kept[0]["entries"])
