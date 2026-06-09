"""Build per-sequence smoke tube JSON dataset from label .txt files.

For each sequence under ``--input-dir/{wildfire,fp}/``:

1. Load detections from labels (5-col GT for wildfire, 6-col YOLO for fp).
2. Build candidate tubes via greedy IoU matching.
3. Select the longest tube; tie-break by non-gap entries.
4. Geometrically interpolate gap bboxes.
5. Apply length / observation filters.
6. Write a JSON file per surviving sequence and a summary file with
   per-split stats and dropped-sequence reasons.

No YOLO inference is performed -- the labels carry everything we need.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from temporal_model.core.labels import load_frame_detections
from temporal_model.core.sequences import is_wf_sequence, list_sequences
from temporal_model.core.tubes import (
    build_tubes,
    interpolate_gaps,
    merge_colocated_tubes,
    select_longest_tube,
)
from temporal_model.core.types import Tube


@dataclass
class DropRecord:
    sequence_id: str
    reason: str


def _serialize_tube(
    *,
    sequence_id: str,
    split: str,
    label: str,
    source: str,
    num_frames: int,
    tube: Tube,
    frame_id_by_idx: dict[int, str],
) -> dict:
    return {
        "sequence_id": sequence_id,
        "split": split,
        "label": label,
        "source": source,
        "num_frames": num_frames,
        "tube": {
            "start_frame": tube.start_frame,
            "end_frame": tube.end_frame,
            "entries": [
                {
                    "frame_idx": e.frame_idx,
                    "frame_id": frame_id_by_idx.get(e.frame_idx, ""),
                    "bbox": [
                        e.detection.cx,
                        e.detection.cy,
                        e.detection.w,
                        e.detection.h,
                    ]
                    if e.detection is not None
                    else None,
                    "is_gap": e.is_gap,
                    "confidence": e.detection.confidence
                    if e.detection is not None
                    else None,
                }
                for e in tube.entries
            ],
        },
    }


def _process_sequence(
    seq_dir: Path,
    *,
    split: str,
    iou_threshold: float,
    max_misses: int,
    min_tube_length: int,
    min_detected_entries: int,
    merge_iomin: float | None,
    merge_prox_factor: float | None,
    merge_max_gap: int | None,
) -> tuple[dict | None, str | None]:
    """Process a single sequence.

    Returns ``(record_or_None, drop_reason_or_None)``.
    """
    is_wf = is_wf_sequence(seq_dir)
    label = "smoke" if is_wf else "fp"
    source = "gt" if is_wf else "yolo"

    if not (seq_dir / "labels").is_dir():
        return None, "no_labels_dir"

    fdets = load_frame_detections(seq_dir)
    if not fdets:
        return None, "no_frames"

    total_dets = sum(len(fd.detections) for fd in fdets)
    if total_dets < min_detected_entries:
        return None, "no_detections"

    tubes = build_tubes(fdets, iou_threshold=iou_threshold, max_misses=max_misses)
    if not tubes:
        return None, "no_tubes"

    # Fuse fragments of the same plume before picking the longest, so the
    # selection sees the merged plume rather than one of its pieces.
    if (
        merge_iomin is not None
        and merge_prox_factor is not None
        and merge_max_gap is not None
    ):
        tubes = merge_colocated_tubes(
            tubes,
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )

    selected = select_longest_tube(tubes)
    assert selected is not None  # tubes is non-empty

    length = selected.end_frame - selected.start_frame + 1
    if length < min_tube_length:
        return None, "too_short"

    n_observed = sum(1 for e in selected.entries if e.detection is not None)
    if n_observed < min_detected_entries:
        return None, "too_few_observed"

    interpolate_gaps(selected)

    frame_id_by_idx = {fd.frame_idx: fd.frame_id for fd in fdets}
    record = _serialize_tube(
        sequence_id=seq_dir.name,
        split=split,
        label=label,
        source=source,
        num_frames=len(fdets),
        tube=selected,
        frame_id_by_idx=frame_id_by_idx,
    )
    return record, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--min-tube-length", type=int, default=4)
    parser.add_argument("--min-detected-entries", type=int, default=2)
    parser.add_argument(
        "--merge-iomin",
        type=float,
        default=None,
        help="Post-build merge: containment threshold. Omit to disable merge.",
    )
    parser.add_argument(
        "--merge-prox-factor",
        type=float,
        default=None,
        help=(
            "Post-build merge: centers-within-N box-sizes factor. "
            "Omit to disable merge."
        ),
    )
    parser.add_argument(
        "--merge-max-gap",
        type=int,
        default=None,
        help=(
            "Post-build merge: max frames between observed spans. "
            "Omit to disable merge."
        ),
    )
    args = parser.parse_args()

    split = args.input_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = list_sequences(args.input_dir)
    written = 0
    by_label: dict[str, int] = {"smoke": 0, "fp": 0}
    dropped: list[DropRecord] = []

    for seq_dir in seq_dirs:
        record, reason = _process_sequence(
            seq_dir,
            split=split,
            iou_threshold=args.iou_threshold,
            max_misses=args.max_misses,
            min_tube_length=args.min_tube_length,
            min_detected_entries=args.min_detected_entries,
            merge_iomin=args.merge_iomin,
            merge_prox_factor=args.merge_prox_factor,
            merge_max_gap=args.merge_max_gap,
        )
        if reason is not None:
            dropped.append(DropRecord(sequence_id=seq_dir.name, reason=reason))
            continue

        out_path = args.output_dir / f"{seq_dir.name}.json"
        out_path.write_text(json.dumps(record, indent=2))
        written += 1
        by_label[record["label"]] += 1

    summary = {
        "split": split,
        "total_sequences": len(seq_dirs),
        "tubes_written": written,
        "by_label": by_label,
        "dropped": [
            {"sequence_id": d.sequence_id, "reason": d.reason} for d in dropped
        ],
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"[{split}] wrote {written}/{len(seq_dirs)} tubes "
        f"(smoke={by_label['smoke']}, fp={by_label['fp']}, "
        f"dropped={len(dropped)})"
    )


if __name__ == "__main__":
    main()
