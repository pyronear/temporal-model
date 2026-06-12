"""Stabilized crop window, recomputed from verbose-response tube entries.

Port of ``core/src/temporal_model/core/stabilize.py`` (the API's verbose
details omit ``stabilized_window``; the viewer's crop panel needs it). Same
policy: union of the tube's observed (non-gap) boxes, falling back to the
union of all available boxes; None when the tube has no usable detection.
Operates on the API's entry dicts rather than core dataclasses.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]  # normalized (cx, cy, w, h)


def union_window(boxes: list[Box]) -> Box:
    """Enclosing box of ``boxes``; raises ValueError on empty input."""
    if not boxes:
        raise ValueError("union_window requires at least one box")
    x0 = min(cx - w / 2 for cx, _, w, _ in boxes)
    y0 = min(cy - h / 2 for _, cy, _, h in boxes)
    x1 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y1 = max(cy + h / 2 for _, cy, _, h in boxes)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def tube_stabilized_window(entries: list[dict]) -> Box | None:
    """Fixed crop window for one verbose-response tube, or None."""
    boxes = [
        (tuple(e["bbox"]) if e.get("bbox") is not None else None, bool(e["is_gap"]))
        for e in entries
    ]
    available = [b for b, _ in boxes if b is not None]
    observed = [b for b, is_gap in boxes if b is not None and not is_gap]
    chosen = observed or available
    return union_window(chosen) if chosen else None
