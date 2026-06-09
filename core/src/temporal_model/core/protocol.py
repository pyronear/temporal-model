"""Temporal model contract: shared types and the ``TemporalModel`` base class.

Vendored from the standalone ``pyrocore`` library so this monorepo is
self-contained. Defines the :class:`Frame` and :class:`TemporalModelOutput`
data types and the :class:`TemporalModel` ABC that all temporal smoke
detection models implement. Uses a template-method pattern:
:meth:`TemporalModel.predict_sequence` wires :meth:`TemporalModel.load_sequence`
(overridable) and :meth:`TemporalModel.predict` (abstract).
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

__all__ = ["Frame", "TemporalModelOutput", "TemporalModel", "parse_timestamp"]


@dataclass
class Frame:
    """A single frame in a temporal sequence.

    Attributes:
        frame_id: Unique identifier, typically the image filename stem.
        image_path: Path to the frame image file.
        timestamp: Capture time, or ``None`` if it cannot be parsed.
    """

    frame_id: str
    image_path: Path
    timestamp: datetime | None = None


@dataclass
class TemporalModelOutput:
    """Output of a temporal model for a single sequence.

    Attributes:
        is_positive: Binary classification decision (``True`` = smoke detected).
        trigger_frame_index: Index of the frame (0-based) where the model
            decided positive, or ``None`` if negative. Time-to-detection
            (TTD) in frames equals this value for a true positive, and
            is interpreted as **delay from the first frame**: TTD = 0
            means instant detection on frame 0; TTD = 1 means one frame
            of delay, etc. Multiply by the per-frame cadence (30s in
            production) to convert to wall-clock delay. Do not compute
            TTD by subtracting frame filename timestamps — they are
            unreliable in the pyro-dataset test set.
        details: Arbitrary model-specific metadata (e.g., tracks, confidence
            scores, attention maps).
    """

    is_positive: bool
    trigger_frame_index: int | None = None
    details: dict = field(default_factory=dict)


_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")


def parse_timestamp(frame_id: str) -> datetime | None:
    """Attempt to extract a timestamp from a Pyronear-style frame ID.

    Expects the frame ID to end with a timestamp segment matching
    ``YYYY-MM-DDTHH-MM-SS`` (e.g., ``adf_site_999_2023-05-23T17-18-31``).

    Returns:
        Parsed :class:`~datetime.datetime`, or ``None`` if parsing fails.
    """
    match = _TIMESTAMP_RE.search(frame_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


class TemporalModel(ABC):
    """Base class for temporal smoke detection models.

    Subclasses **must** implement :meth:`predict`.  They **may** override
    :meth:`load_sequence` to customise how frame image paths are converted
    into :class:`Frame` objects (e.g., to add cached detections or parse
    timestamps with a different naming convention).

    Callers use :meth:`predict_sequence` as the single entry point.

    Example::

        class MyModel(TemporalModel):
            def predict(self, frames: list[Frame]) -> TemporalModelOutput:
                # model logic here
                return TemporalModelOutput(is_positive=True, trigger_frame_index=4)

        model = MyModel()
        output = model.predict_sequence(sorted_frame_paths)
    """

    def load_sequence(self, frames: list[Path]) -> list[Frame]:
        """Load a sequence of frame image paths into :class:`Frame` objects.

        The default implementation builds a :class:`Frame` for each path with:

        - ``frame_id`` set to the filename stem,
        - ``image_path`` set to the input path,
        - ``timestamp`` parsed from the Pyronear filename convention
          (``<prefix>_<YYYY-MM-DDTHH-MM-SS>``), falling back to ``None``.

        Override this method to parse timestamps differently, attach cached
        YOLO detections, or perform other custom loading.

        Args:
            frames: Temporally ordered list of frame image paths.

        Returns:
            List of :class:`Frame` objects in the same order.
        """
        return [
            Frame(
                frame_id=p.stem,
                image_path=p,
                timestamp=parse_timestamp(p.stem),
            )
            for p in frames
        ]

    @abstractmethod
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run temporal model logic on a loaded sequence.

        This is the method each subclass must implement.

        Args:
            frames: Temporally ordered list of :class:`Frame` objects,
                as returned by :meth:`load_sequence`.

        Returns:
            :class:`TemporalModelOutput` with the classification decision
            and optional timing/details.
        """
        ...

    def predict_sequence(self, frames: list[Path]) -> TemporalModelOutput:
        """Main entry point: load frame images then predict.

        Calls :meth:`load_sequence` followed by :meth:`predict`.  This
        method should generally not be overridden.

        Args:
            frames: Temporally ordered list of frame image paths.

        Returns:
            :class:`TemporalModelOutput` with classification decision and timing.
        """
        loaded = self.load_sequence(frames)
        return self.predict(loaded)
