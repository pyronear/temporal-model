"""Run BboxTubeTemporalModel.predict() over a sequence store, timing each stage."""

import logging
from pathlib import Path

import pandas as pd
import torch

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.stage_timer import StageTimer

from .dataset import BenchSequence, iter_sequences

logger = logging.getLogger(__name__)

STAGES = ("pad", "yolo", "tubes", "crop", "vit", "trigger")


def resolve_device(requested: str) -> str:
    """Map ``auto`` to cuda-if-available else cpu; pass anything else through."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _one_rep(model: BboxTubeTemporalModel, seq: BenchSequence, device: str) -> dict:
    timer = StageTimer(device=device)
    output = model.predict(seq.frames, timer=timer)
    timings = timer.as_dict()
    row = {
        "key": seq.key,
        "label": seq.label,
        "frame_count": seq.frame_count,
        "n_kept_tubes": len(output.details.get("tubes", {}).get("kept", [])),
        "is_positive": output.is_positive,
    }
    for stage in STAGES:
        row[f"{stage}_ms"] = timings.get(stage, 0.0)
    row["total_ms"] = sum(timings.get(s, 0.0) for s in STAGES)
    return row


def run_core(
    store_dir: Path,
    model_path: Path,
    *,
    device: str = "auto",
    reps: int = 5,
    warmup: int = 3,
    limit: int | None = None,
) -> pd.DataFrame:
    """Benchmark predict() over every sequence; one row per (sequence, rep)."""
    device = resolve_device(device)
    model = BboxTubeTemporalModel.from_package(model_path, device=device)

    sequences = list(iter_sequences(store_dir))
    if limit is not None:
        sequences = sequences[:limit]
    if not sequences:
        raise SystemExit(f"no sequences found under {store_dir}")

    logger.info("warming up on %d sequences", min(warmup, len(sequences)))
    for seq in sequences[:warmup]:
        model.predict(seq.frames)

    rows: list[dict] = []
    for i, seq in enumerate(sequences):
        for rep in range(reps):
            try:
                row = _one_rep(model, seq, device)
                row["rep"] = rep
                row["failed"] = False
            except Exception as exc:  # noqa: BLE001 — record + continue
                logger.warning("sequence %s failed: %s", seq.key, exc)
                row = {"key": seq.key, "rep": rep, "failed": True}
            rows.append(row)
        if (i + 1) % 25 == 0:
            logger.info("benchmarked %d/%d sequences", i + 1, len(sequences))

    return pd.DataFrame(rows)
