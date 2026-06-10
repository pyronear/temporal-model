"""Smoke tube construction via greedy IoU matching across frames.

A *tube* is a chain of YOLO detections across consecutive frames that
correspond to the same spatial smoke region.  Tubes bridge per-frame
detections and the LSTM's need for temporally-ordered features of the
same entity.
"""

from collections.abc import Callable

from .types import Detection, FrameDetections, Tube, TubeEntry

__all__ = [
    "compute_iou",
    "match_detections",
    "build_tubes",
    "interpolate_gaps",
    "select_longest_tube",
    "tube_from_record",
    "merge_colocated_tubes",
    "tube_intersects_roi",
    "validate_roi",
]


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections.

    Both detections use normalised center-based coordinates
    (cx, cy, w, h) in [0, 1].

    Returns:
        IoU value in [0, 1].  Returns 0.0 when the union area is zero.
    """
    a_x1 = det_a.cx - det_a.w / 2
    a_y1 = det_a.cy - det_a.h / 2
    a_x2 = det_a.cx + det_a.w / 2
    a_y2 = det_a.cy + det_a.h / 2

    b_x1 = det_b.cx - det_b.w / 2
    b_y1 = det_b.cy - det_b.h / 2
    b_x2 = det_b.cx + det_b.w / 2
    b_y2 = det_b.cy + det_b.h / 2

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = det_a.w * det_a.h
    b_area = det_b.w * det_b.h
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_detections(
    prev_dets: list[Detection],
    curr_dets: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of detections between consecutive frames.

    Args:
        prev_dets: Detections from the previous frame.
        curr_dets: Detections from the current frame.
        iou_threshold: Minimum IoU required to consider a match.

    Returns:
        List of ``(prev_idx, curr_idx, iou)`` tuples.
    """
    if not prev_dets or not curr_dets:
        return []

    pairs: list[tuple[float, int, int]] = []
    for i, prev in enumerate(prev_dets):
        for j, curr in enumerate(curr_dets):
            iou = compute_iou(prev, curr)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(key=lambda x: -x[0])
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou_val, i, j in pairs:
        if i not in matched_prev and j not in matched_curr:
            matches.append((i, j, iou_val))
            matched_prev.add(i)
            matched_curr.add(j)

    return matches


def build_tubes(
    frame_detections: list[FrameDetections],
    iou_threshold: float = 0.2,
    max_misses: int = 2,
) -> list[Tube]:
    """Build smoke tubes from per-frame detections.

    Tubes are constructed by greedy IoU matching: each detection in
    frame *t* is matched to the closest (by IoU) active tube from
    frame *t-1*.  Unmatched detections start new tubes; unmatched
    tubes accumulate misses and are terminated after *max_misses*
    consecutive gaps.

    Args:
        frame_detections: Temporally ordered per-frame detection lists.
        iou_threshold: Minimum IoU for matching a detection to a tube.
        max_misses: Maximum consecutive gap frames before a tube is
            terminated.

    Returns:
        List of :class:`Tube` objects covering the full sequence.
    """
    if not frame_detections:
        return []

    active_tubes: list[Tube] = []
    finished_tubes: list[Tube] = []
    next_tube_id = 0
    # Track consecutive misses per active tube
    consecutive_misses: list[int] = []

    for frame in frame_detections:
        frame_idx = frame.frame_idx
        curr_dets = frame.detections

        if active_tubes and curr_dets:
            # Get last known detection for each active tube
            last_dets = [_last_detection(tube) for tube in active_tubes]
            matches = match_detections(last_dets, curr_dets, iou_threshold)
            matched_tube_idxs = {m[0] for m in matches}
            matched_det_idxs = {m[1] for m in matches}

            # Extend matched tubes
            for tube_idx, det_idx, _ in matches:
                tube = active_tubes[tube_idx]
                # Fill gap entries for any missed frames
                _fill_gaps(tube, frame_idx)
                tube.entries.append(
                    TubeEntry(frame_idx=frame_idx, detection=curr_dets[det_idx])
                )
                tube.end_frame = frame_idx
                consecutive_misses[tube_idx] = 0

            # Increment misses for unmatched tubes
            for i in range(len(active_tubes)):
                if i not in matched_tube_idxs:
                    consecutive_misses[i] += 1

            # Start new tubes for unmatched detections
            for j, det in enumerate(curr_dets):
                if j not in matched_det_idxs:
                    new_tube = Tube(
                        tube_id=next_tube_id,
                        entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                    )
                    next_tube_id += 1
                    active_tubes.append(new_tube)
                    consecutive_misses.append(0)
        else:
            # Increment misses for all active tubes (no detections this frame)
            for i in range(len(active_tubes)):
                consecutive_misses[i] += 1

            # Start new tubes for all detections
            for det in curr_dets:
                new_tube = Tube(
                    tube_id=next_tube_id,
                    entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                )
                next_tube_id += 1
                active_tubes.append(new_tube)
                consecutive_misses.append(0)

        # Prune tubes that exceeded max_misses
        surviving_tubes: list[Tube] = []
        surviving_misses: list[int] = []
        for tube, misses in zip(active_tubes, consecutive_misses, strict=True):
            if misses > max_misses:
                finished_tubes.append(tube)
            else:
                surviving_tubes.append(tube)
                surviving_misses.append(misses)
        active_tubes = surviving_tubes
        consecutive_misses = surviving_misses

    # All remaining active tubes are finished
    finished_tubes.extend(active_tubes)

    # Sort by tube_id for deterministic output
    finished_tubes.sort(key=lambda t: t.tube_id)
    return finished_tubes


def _last_detection(tube: Tube) -> Detection:
    """Return the most recent non-gap detection in a tube."""
    for entry in reversed(tube.entries):
        if entry.detection is not None:
            return entry.detection
    raise ValueError(f"Tube {tube.tube_id} has no detections")


def _fill_gaps(tube: Tube, target_frame_idx: int) -> None:
    """Insert gap entries (detection=None) for missed frames."""
    if not tube.entries:
        return
    last_frame = tube.entries[-1].frame_idx
    for gap_idx in range(last_frame + 1, target_frame_idx):
        tube.entries.append(TubeEntry(frame_idx=gap_idx, detection=None))


def interpolate_gaps(tube: Tube) -> Tube:
    """Fill gap entries with a geometrically-interpolated bbox.

    For each entry whose ``detection`` is ``None``:

    * **Interior gap** (observed dets on both sides): linearly interpolate
      ``(cx, cy, w, h)`` between the nearest observed detection before and
      after, using the entry's index as the position parameter.
    * **Boundary gap** (no observation on one side): repeat the nearest
      observed detection on the other side.

    Synthesized detections always carry ``confidence=0.0``. The returned
    tube has ``is_gap=True`` flags on every previously-empty entry.

    Observed entries are left untouched.

    Args:
        tube: Tube whose gap entries (``detection=None``) need filling.

    Returns:
        The same tube object, mutated in place. Returned for chaining.
    """
    observed = [
        (i, e.detection) for i, e in enumerate(tube.entries) if e.detection is not None
    ]
    if not observed:
        return tube

    for i, entry in enumerate(tube.entries):
        if entry.detection is not None:
            continue

        before = next(
            ((j, d) for j, d in reversed(observed) if j < i),
            None,
        )
        after = next(
            ((j, d) for j, d in observed if j > i),
            None,
        )

        if before is not None and after is not None:
            j_b, d_b = before
            j_a, d_a = after
            t = (i - j_b) / (j_a - j_b)
            cx = d_b.cx + t * (d_a.cx - d_b.cx)
            cy = d_b.cy + t * (d_a.cy - d_b.cy)
            w = d_b.w + t * (d_a.w - d_b.w)
            h = d_b.h + t * (d_a.h - d_b.h)
            class_id = d_b.class_id
        elif before is not None:
            d = before[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id
        else:
            assert after is not None
            d = after[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id

        entry.detection = Detection(
            class_id=class_id,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            confidence=0.0,
        )
        entry.is_gap = True

    return tube


def select_longest_tube(tubes: list[Tube]) -> Tube | None:
    """Pick the single longest tube from a list.

    Length is measured as ``end_frame - start_frame + 1`` (so gaps count
    toward length). Ties are broken by the number of non-gap entries --
    the tube with more real observations wins. If still tied, the first
    in the input order wins.

    Args:
        tubes: Candidate tubes.

    Returns:
        The selected tube, or ``None`` if ``tubes`` is empty.
    """
    if not tubes:
        return None

    def _key(tube: Tube) -> tuple[int, int]:
        length = tube.end_frame - tube.start_frame + 1
        n_observed = sum(1 for e in tube.entries if e.detection is not None)
        return (length, n_observed)

    return max(tubes, key=_key)


def tube_from_record(record: dict) -> Tube:
    """Rebuild a :class:`Tube` from a tube JSON record.

    Inverse of ``_serialize_tube`` in ``scripts/build_tubes.py``. Pure
    function; no I/O.

    Entries with ``bbox=None`` are reconstructed with ``detection=None``
    (pre-interpolation gap shape). Otherwise a :class:`Detection` is
    built from the bbox + confidence; ``confidence=None`` falls back to
    ``0.0``.

    Args:
        record: Parsed tube record. Only the ``tube`` sub-object is
            consulted; other top-level keys are ignored.

    Returns:
        A :class:`Tube` with ``tube_id=0`` (the on-disk dataset is
        single-tube-per-sequence so the id is informational only).
    """
    t = record["tube"]
    entries: list[TubeEntry] = []
    for e in t["entries"]:
        bbox = e["bbox"]
        if bbox is None:
            det: Detection | None = None
        else:
            det = Detection(
                class_id=0,
                cx=bbox[0],
                cy=bbox[1],
                w=bbox[2],
                h=bbox[3],
                confidence=e["confidence"] if e["confidence"] is not None else 0.0,
            )
        entries.append(
            TubeEntry(
                frame_idx=e["frame_idx"],
                detection=det,
                is_gap=e["is_gap"],
            )
        )
    return Tube(
        tube_id=0,
        entries=entries,
        start_frame=t["start_frame"],
        end_frame=t["end_frame"],
    )


# ── post-hoc co-located merge ───────────────────────────────────────────────
#
# Fragments-of-the-same-plume merge that runs after build_tubes. Two tubes are
# linked when temporally close AND, at the frames where they are nearest in
# time, the smaller is mostly inside the larger (IoMin) OR their centers are
# within ~merge_prox_factor box-sizes (scale-relative, so tiny boxes don't
# chain across many widths). Merging is connected-components under that
# relation; per-frame collisions on overlap keep the higher-ranked box
# (largest area, then highest confidence, then larger cx — deterministic).

Observed = list[tuple[int, Detection]]


def merge_colocated_tubes(
    tubes: list[Tube],
    *,
    merge_iomin: float,
    merge_prox_factor: float,
    merge_max_gap: int,
) -> list[Tube]:
    """Merge tubes that are fragments of the same plume.

    Args:
        tubes: candidate tubes (typically from :func:`build_tubes` after the
            inference filter).
        merge_iomin: containment threshold — link if the smaller box's
            intersection-over-min-area with the larger is at least this.
        merge_prox_factor: proximity factor — link if centers are within this
            many box-sizes (scale-relative).
        merge_max_gap: temporal gap, in frames, allowed between two observed
            spans for them to still be considered the same plume.

    Returns:
        Merged tubes, sorted by ``(start_frame, end_frame)`` with sequentially
        re-assigned ``tube_id``s. Gap entries (``detection=None``) are inserted
        for unobserved frames between observed ones.
    """
    observed = [_observed(t) for t in tubes]

    def related(i: int, j: int) -> bool:
        return _same_plume(
            observed[i],
            observed[j],
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )

    components = _connected_components(len(tubes), related)
    merged = [_combine(c, observed) for c in components]
    merged = [t for t in merged if t is not None]
    merged.sort(key=lambda t: (t.start_frame, t.end_frame))
    for tube_id, tube in enumerate(merged):
        tube.tube_id = tube_id
    return merged


# ── the "same plume" relation ───────────────────────────────────────────────


def _same_plume(
    a: Observed,
    b: Observed,
    *,
    merge_iomin: float,
    merge_prox_factor: float,
    merge_max_gap: int,
) -> bool:
    if not a or not b:
        return False
    if _time_gap(a, b) > merge_max_gap:
        return False
    det_a, det_b = _closest_in_time(a, b)
    return _same_box(
        det_a, det_b, merge_iomin=merge_iomin, merge_prox_factor=merge_prox_factor
    )


def _time_gap(a: Observed, b: Observed) -> int:
    """Frames separating the two observed spans (0 when they overlap)."""
    a_start, a_end = a[0][0], a[-1][0]
    b_start, b_end = b[0][0], b[-1][0]
    if b_start > a_end:
        return b_start - a_end
    if a_start > b_end:
        return a_start - b_end
    return 0


def _closest_in_time(a: Observed, b: Observed) -> tuple[Detection, Detection]:
    best = min(
        ((abs(fa - fb), da, db) for fa, da in a for fb, db in b),
        key=lambda candidate: candidate[0],
    )
    return best[1], best[2]


def _same_box(
    a: Detection,
    b: Detection,
    *,
    merge_iomin: float,
    merge_prox_factor: float,
) -> bool:
    if _iou_min(a, b) >= merge_iomin:
        return True
    box_size = max(a.w, a.h, b.w, b.h)
    return _center_distance(a, b) <= merge_prox_factor * box_size


def _iou_min(a: Detection, b: Detection) -> float:
    """Intersection over the smaller box's area (containment)."""
    ax1, ay1, ax2, ay2 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1, bx2, by2 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


# ── grouping and rebuilding ─────────────────────────────────────────────────


def _connected_components(
    n: int, related: Callable[[int, int], bool]
) -> list[list[int]]:
    """Partition indices ``range(n)`` into components linked by ``related``."""
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if related(i, j):
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _combine(members: list[int], observed: list[Observed]) -> Tube | None:
    """Fuse a component's fragments into one tube (largest box wins on overlap)."""
    best_by_frame: dict[int, Detection] = {}
    for m in members:
        for frame_idx, det in observed[m]:
            best = best_by_frame.get(frame_idx)
            if best is None or _box_rank(det) > _box_rank(best):
                best_by_frame[frame_idx] = det
    if not best_by_frame:
        return None
    start, end = min(best_by_frame), max(best_by_frame)
    entries = [
        TubeEntry(frame_idx=f, detection=best_by_frame.get(f))
        for f in range(start, end + 1)
    ]
    return Tube(tube_id=0, entries=entries, start_frame=start, end_frame=end)


def _observed(tube: Tube) -> Observed:
    return [(e.frame_idx, e.detection) for e in tube.entries if e.detection is not None]


def _area(det: Detection) -> float:
    return det.w * det.h


def _box_rank(det: Detection) -> tuple[float, float, float]:
    """Pick order among boxes on the same frame: larger area, then higher
    confidence, then larger cx — a total order, so the choice is independent of
    the input tube order (no silent nondeterminism on equal-area ties)."""
    return (_area(det), det.confidence, det.cx)


def validate_roi(roi: tuple[float, float, float, float]) -> None:
    """Validate a normalized ROI rectangle, raising ``ValueError`` if invalid.

    Single source of truth for ROI validity — the API request validator and
    ``BboxTubeTemporalModel.predict`` both call this, so the rules cannot
    drift between layers. Valid: 4 values in [0, 1] with x_min < x_max and
    y_min < y_max.
    """
    if len(roi) != 4:
        raise ValueError(
            f"invalid roi {roi!r}: expected 4 values (x_min, y_min, x_max, y_max)"
        )
    x_min, y_min, x_max, y_max = roi
    if not all(0.0 <= c <= 1.0 for c in roi) or x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"invalid roi {roi!r}: expected normalized "
            "(x_min, y_min, x_max, y_max) with x_min < x_max and y_min < y_max"
        )


def tube_intersects_roi(tube: Tube, roi: tuple[float, float, float, float]) -> bool:
    """True if any real (non-gap) detection bbox overlaps the ROI rectangle.

    ``roi`` is ``(x_min, y_min, x_max, y_max)`` normalized to [0, 1];
    touching edges count as overlap. Gap entries are synthetic (interpolated)
    and do not count.
    """
    x_min, y_min, x_max, y_max = roi
    for entry in tube.entries:
        if entry.is_gap or entry.detection is None:
            continue
        d = entry.detection
        if (
            d.cx - d.w / 2 <= x_max
            and d.cx + d.w / 2 >= x_min
            and d.cy - d.h / 2 <= y_max
            and d.cy + d.h / 2 >= y_min
        ):
            return True
    return False
