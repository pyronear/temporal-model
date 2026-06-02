"""Tests for temporal_model.core.types."""

from datetime import datetime

from temporal_model.core.types import (
    Detection,
    FrameDetections,
    Tube,
    TubeEntry,
)


class TestDetection:
    def test_construction(self):
        det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
        assert det.class_id == 0
        assert det.cx == 0.5
        assert det.confidence == 0.9

    def test_area(self):
        det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3, confidence=0.5)
        assert abs(det.w * det.h - 0.06) < 1e-9


class TestFrameDetections:
    def test_construction(self):
        det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
        fd = FrameDetections(
            frame_idx=0,
            frame_id="frame_000",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            detections=[det],
        )
        assert fd.frame_idx == 0
        assert len(fd.detections) == 1

    def test_none_timestamp(self):
        fd = FrameDetections(
            frame_idx=0, frame_id="frame_000", timestamp=None, detections=[]
        )
        assert fd.timestamp is None


class TestTubeEntry:
    def test_with_detection(self):
        det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
        entry = TubeEntry(frame_idx=0, detection=det)
        assert entry.detection is not None

    def test_gap(self):
        entry = TubeEntry(frame_idx=2)
        assert entry.detection is None


class TestTube:
    def test_construction(self):
        tube = Tube(tube_id=0, start_frame=0, end_frame=5)
        assert tube.tube_id == 0
        assert tube.entries == []

    def test_with_entries(self):
        det = Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.9)
        entries = [
            TubeEntry(frame_idx=0, detection=det),
            TubeEntry(frame_idx=1),  # gap
            TubeEntry(frame_idx=2, detection=det),
        ]
        tube = Tube(tube_id=0, entries=entries, start_frame=0, end_frame=2)
        assert len(tube.entries) == 3
        assert tube.entries[1].detection is None
