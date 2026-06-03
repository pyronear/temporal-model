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


def collect_pipeline_records(*, model: Any, raw_dir: Path) -> list[dict]:
    """Run ``model.predict_sequence`` on every labelled sequence under ``raw_dir``.

    Args:
        model: A ``BboxTubeTemporalModel`` (or duck-type) with
            ``predict_sequence(frame_paths) -> TemporalModelOutput`` whose
            ``.details["tubes"]["kept"]`` carries the tube structure.
        raw_dir: ``{wildfire,fp}/<seq>/images/*.jpg`` tree.
    """
    records: list[dict] = []
    for label, seq_name, frame_paths in _iter_labelled_sequences(raw_dir):
        out = model.predict_sequence(frame_paths)
        kept = out.details.get("tubes", {}).get("kept", [])
        records.append({"label": label, "sequence": seq_name, "kept_tubes": kept})
    return records
