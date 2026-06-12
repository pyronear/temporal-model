"""Reconstruct the exact /predict call alert-api made for a sequence.

Line-for-line port of pyro-api's ``validation._sequence_frames_and_roi``
(pyro-api ``src/app/services/validation.py``) and its bbox parsing
(``src/app/api/api_v1/endpoints/detections.py`` + ``schemas/detections.py``):
distinct bucket_keys oldest first, truncated to the most recent MAX_FRAMES;
ROI = union envelope of the kept detections' PRIMARY bboxes (the first box in
each detection's bbox string), clamped to [0, 1], None when nothing parses or
the envelope is degenerate. Keep this in sync with pyro-api — parity is the
whole point of replay.

Known, accepted limit (spec): detections that arrived after the last
production scoring can shift the reconstruction; the replay_matches check
makes that visible.
"""

from __future__ import annotations

import re
from ast import literal_eval

from temporal_model.monitor.store import FrameMeta

# pyro-api src/app/services/temporal.py
MIN_FRAMES = 4
MAX_FRAMES = 10

# pyro-api src/app/schemas/detections.py (verbatim)
FLOAT_PATTERN = r"(0?\.[0-9]{1,3}|0|1)"
BOX_PATTERN = (
    rf"\({FLOAT_PATTERN},{FLOAT_PATTERN},{FLOAT_PATTERN},"
    rf"{FLOAT_PATTERN},{FLOAT_PATTERN}\)"
)


def extract_bbox_strings(bboxes: str) -> list[str]:
    return [match.group(0) for match in re.finditer(BOX_PATTERN, bboxes)]


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float, float] | None:
    """Parse one '(xmin,ymin,xmax,ymax,conf)' string; None when malformed.

    pyro-api raises HTTP 422 here; the worker catches it and skips the bbox —
    returning None reproduces the skip without the exception plumbing.
    """
    try:
        bbox = literal_eval(bbox_str)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(bbox, tuple) or len(bbox) != 5:
        return None
    return bbox


def frames_and_roi(
    frames: list[FrameMeta], last_n: int | None = MAX_FRAMES
) -> tuple[int, list[str], list[float] | None]:
    """(total_distinct, kept_frame_keys, roi_xyxyn) for a stored sequence.

    ``frames`` must be ordered by created_at ascending (the store writes them
    that way), matching the worker's ``order_by="created_at"`` fetch.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    corners_by_frame: dict[str, list[tuple[float, float, float, float]]] = {}
    for f in frames:
        if f.bucket_key not in seen:
            seen.add(f.bucket_key)
            ordered.append(f.bucket_key)
        bbox_strs = extract_bbox_strings(f.bbox)
        if bbox_strs:
            parsed = parse_bbox(bbox_strs[0])
            if parsed is not None:
                xmin, ymin, xmax, ymax, _ = parsed
                corners_by_frame.setdefault(f.bucket_key, []).append(
                    (xmin, ymin, xmax, ymax)
                )
    total = len(ordered)
    kept = ordered if last_n is None or total <= last_n else ordered[-last_n:]
    corners = [c for fr in kept for c in corners_by_frame.get(fr, [])]
    if not corners:
        return total, kept, None
    roi = [
        max(0.0, min(c[0] for c in corners)),
        max(0.0, min(c[1] for c in corners)),
        min(1.0, max(c[2] for c in corners)),
        min(1.0, max(c[3] for c in corners)),
    ]
    if not (roi[0] < roi[2] and roi[1] < roi[3]):
        return total, kept, None
    return total, kept, roi
