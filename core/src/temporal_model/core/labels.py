"""Label-file and tube-record loading for the pyro-dataset layout."""

import json
from pathlib import Path

from .protocol import parse_timestamp
from .sequences import get_sorted_frames
from .types import Detection, FrameDetections

__all__ = [
    "load_detections",
    "load_frame_detections",
    "load_tube_record",
]


def load_detections(sequence_dir: Path, frame_id: str) -> list[Detection]:
    """Read a YOLO-format label file as :class:`Detection` objects.

    Supports both formats found in the Pyronear dataset:

    * **5-col** ``class cx cy w h`` -- wildfire ground-truth annotations.
      ``confidence`` is set to ``1.0``.
    * **6-col** ``class cx cy w h conf`` -- false-positive YOLO predictions.
      ``confidence`` is read from the last column.

    Malformed lines (wrong column count, non-numeric values) are silently
    skipped.

    Args:
        sequence_dir: Path to the sequence directory (contains ``labels/``).
        frame_id: Frame filename stem.

    Returns:
        List of detections in file order. Empty list if the file is missing
        or empty.
    """
    label_path = sequence_dir / "labels" / f"{frame_id}.txt"
    if not label_path.is_file():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []
    dets: list[Detection] = []
    for line in content.split("\n"):
        parts = line.strip().split()
        try:
            if len(parts) == 5:
                class_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:5])
                confidence = 1.0
            elif len(parts) == 6:
                class_id = int(parts[0])
                cx, cy, w, h, confidence = (float(p) for p in parts[1:6])
            else:
                continue
        except ValueError:
            continue
        dets.append(
            Detection(
                class_id=class_id,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                confidence=confidence,
            )
        )
    return dets


def load_frame_detections(sequence_dir: Path) -> list[FrameDetections]:
    """Load all per-frame detections for a sequence in temporal order.

    Iterates frames returned by
    :func:`~temporal_model.core.sequences.get_sorted_frames` and reads the
    corresponding label file via :func:`load_detections`.

    Args:
        sequence_dir: Path to the sequence directory.

    Returns:
        Ordered list of :class:`FrameDetections`, one per image. Frames
        with no labels yield an entry with an empty ``detections`` list.
    """
    frame_paths = get_sorted_frames(sequence_dir)
    return [
        FrameDetections(
            frame_idx=idx,
            frame_id=fpath.stem,
            timestamp=parse_timestamp(fpath.stem),
            detections=load_detections(sequence_dir, fpath.stem),
        )
        for idx, fpath in enumerate(frame_paths)
    ]


def load_tube_record(path: Path) -> dict:
    """Read+parse a tube JSON file.

    Trivial wrapper around :func:`json.loads`; exists so callers
    (scripts, notebooks) have a single named entry point for tube I/O.

    Args:
        path: Path to a tube ``.json`` file produced by
            ``scripts/build_tubes.py``.

    Returns:
        The parsed record as a plain dict.
    """
    return json.loads(path.read_text())
