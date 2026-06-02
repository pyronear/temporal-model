"""Data types for the bbox-tube temporal smoke detection model."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Detection:
    """A single YOLO detection with normalized center-based coordinates.

    All spatial values are in [0, 1] relative to the image dimensions.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


@dataclass
class FrameDetections:
    """All detections from a single frame."""

    frame_idx: int
    frame_id: str
    timestamp: datetime | None
    detections: list[Detection]


@dataclass
class TubeEntry:
    """A single entry in a smoke tube.

    ``is_gap`` flags entries whose ``detection`` was not observed by the
    detector and was instead filled in by gap interpolation. After
    interpolation, gap entries always have a ``Detection`` (lerped bbox,
    confidence=0.0); pre-interpolation gaps have ``detection=None``.
    """

    frame_idx: int
    detection: Detection | None = None
    is_gap: bool = False


@dataclass
class Tube:
    """A smoke tube: a chain of detections across frames for the same region.

    Tubes are constructed by greedy IoU matching across consecutive frames.
    """

    tube_id: int
    entries: list[TubeEntry] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0


@dataclass
class SequenceFeatures:
    """Precomputed features for a single sequence, saved as .pt/.json."""

    sequence_id: str
    num_frames: int
    num_detections: int
    tubes: list[Tube]
    is_positive: bool | None = None
    features_path: Path | None = None
    metadata_path: Path | None = None
