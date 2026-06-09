"""Tests for the pure detection-injection seam on BboxTubeTemporalModel."""

from unittest.mock import MagicMock

# Reuse the shared fixtures/helpers from the edge-case suite.
from test_model_edge_cases import (  # type: ignore[import-not-found]
    TEST_CONFIG,
    _fake_yolo_factory,
    red_frames,
    tiny_classifier,
)

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.types import FrameDetections

__all__ = ["red_frames", "tiny_classifier"]  # keep fixtures importable


def test_detect_returns_one_framedetections_per_frame(red_frames, tiny_classifier):
    per_frame = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    dets = model.detect(red_frames)

    assert [d.frame_id for d in dets] == [f.frame_id for f in red_frames]
    assert [d.frame_idx for d in dets] == list(range(len(red_frames)))
    assert all(len(d.detections) == 1 for d in dets)


def test_injection_parity_matches_full_detection(red_frames, tiny_classifier):
    per_frame = [[(0.5, 0.5, 0.2, 0.2, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    baseline = model.predict(frames=red_frames)
    cached = {fd.frame_id: fd for fd in model.detect(red_frames)}
    injected = model.predict(frames=red_frames, frame_detections=cached)

    assert injected == baseline


def test_injection_detects_only_misses(red_frames, tiny_classifier):
    # All but the last frame are pre-supplied; YOLO must run on the last only.
    cached = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
        )
        for i, f in enumerate(red_frames[:-1])
    }
    yolo = _fake_yolo_factory([[]])  # asserts it is called with exactly 1 frame
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    model.predict(frames=red_frames, frame_detections=cached)

    yolo.predict.assert_called_once()
    called_paths = yolo.predict.call_args[0][0]
    assert called_paths == [str(red_frames[-1].image_path)]


def test_injection_all_cached_skips_yolo(red_frames, tiny_classifier):
    cached = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
        )
        for i, f in enumerate(red_frames)
    }
    yolo = MagicMock()
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
    )

    model.predict(frames=red_frames, frame_detections=cached)

    yolo.predict.assert_not_called()
