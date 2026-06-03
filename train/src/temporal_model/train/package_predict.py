"""Run the full YOLO + tracking + classifier pipeline at package time.

Produces the labelled per-tube records that ``logistic_calibrator_fit.fit`` and
threshold calibration consume. Bypasses the ``.zip`` — the YOLO model,
classifier, and config are already in memory at packaging time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from temporal_model.core.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
)


def _iter_labelled_sequences(raw_dir: Path) -> Iterator[tuple[str, str, list[Path]]]:
    """Yield ``(label, sequence_name, frame_paths)`` for every sequence."""
    for seq_dir in list_sequences(raw_dir):
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            continue
        label = "smoke" if is_wf_sequence(seq_dir) else "fp"
        yield label, seq_dir.name, frame_paths


def collect_pipeline_records(
    *, model: Any, raw_dir: Path, log_every: int = 200
) -> list[dict]:
    """Run ``model.predict_sequence`` on every labelled sequence under ``raw_dir``.

    Args:
        model: A ``BboxTubeTemporalModel`` (or duck-type) with
            ``predict_sequence(frame_paths) -> TemporalModelOutput`` whose
            ``.details["tubes"]["kept"]`` carries the tube structure.
        raw_dir: ``{wildfire,fp}/<seq>/images/*.jpg`` tree.
        log_every: Print a progress line every N sequences (0 disables). This is
            the slow pass, so progress is emitted to stdout for tracking.
    """
    total = len(list_sequences(raw_dir))
    records: list[dict] = []
    for i, (label, seq_name, frame_paths) in enumerate(
        _iter_labelled_sequences(raw_dir), start=1
    ):
        out = model.predict_sequence(frame_paths)
        kept = out.details.get("tubes", {}).get("kept", [])
        records.append({"label": label, "sequence": seq_name, "kept_tubes": kept})
        if log_every and i % log_every == 0:
            print(
                f"[package_predict] {raw_dir.name}: {i}/{total} sequences", flush=True
            )
    return records
