"""Score stored sequences with the temporal model and bucket them by threshold.

Pure helpers (``sequence_score``, ``bucket_for``) hold the triage policy and are
unit-tested without a model; ``score_sequences`` wires the loaded model over the
store. The triage score is the largest calibrated per-tube probability — the
same quantity ``eval`` reports as ``probability`` — thresholded independently of
the model's own keep/discard rule.

With several models (an ensemble), each scores every sequence and the triage
score is the highest of them: a sequence goes to Unlabel only when *every* model
is below the threshold, so one dissenting model is enough to send it to a human.
Sequences the models disagree on are the most informative to annotate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temporal_model.triage.store import (
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    read_meta,
)

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 250  # log a heartbeat every N scored sequences


@dataclass
class ScoredSequence:
    key: str
    sequence_id: int
    score: float
    bucket: str  # "review" | "unlabeled"
    meta: SequenceMeta
    details: dict
    trigger_frame_index: int | None
    frame_paths: list[Path]
    model_scores: list[float] = field(default_factory=list)  # one per model

    def n_models_review(self, threshold: float) -> int:
        """How many models alone would send this sequence to review."""
        return sum(s >= threshold for s in self.model_scores)


def sequence_score(details: dict[str, Any] | None) -> float:
    """Largest calibrated probability across kept tubes; 0.0 if none."""
    kept = (details or {}).get("tubes", {}).get("kept", [])
    probs = [t.get("probability") for t in kept if t.get("probability") is not None]
    return max(probs) if probs else 0.0


def bucket_for(score: float, threshold: float) -> str:
    """'review' for high scorers (>= threshold), else 'unlabeled'."""
    return "review" if score >= threshold else "unlabeled"


def score_sequences(
    models, store_dir: Path, *, threshold: float
) -> tuple[list[ScoredSequence], list[dict]]:
    """Run predict() over every stored sequence; classify each by threshold.

    ``models`` is one model or a list (ensemble); see the module docstring.
    """
    models = models if isinstance(models, list) else [models]
    seq_dirs = list(iter_sequence_dirs(store_dir))
    total = len(seq_dirs)
    start = time.monotonic()
    scored: list[ScoredSequence] = []
    dropped: list[dict] = []
    for i, seq_dir in enumerate(seq_dirs, 1):
        meta = read_meta(seq_dir)
        frame_paths = [seq_dir / f.file for f in meta.frames]
        if not frame_paths:
            dropped.append({"sequence_id": meta.key, "reason": "no_images"})
            continue
        frames = build_frames(seq_dir, meta)
        try:
            outputs = [m.predict(frames, compute_trigger=True) for m in models]
        except Exception as exc:  # noqa: BLE001 — record + continue
            logger.warning("predict failed for %s: %s", meta.key, exc)
            dropped.append({"sequence_id": meta.key, "reason": "predict_failed"})
            continue
        model_scores = [sequence_score(o.details) for o in outputs]
        score = max(model_scores)
        output = outputs[model_scores.index(score)]  # show the most alarmed model
        scored.append(
            ScoredSequence(
                key=meta.key,
                sequence_id=meta.sequence_id,
                score=score,
                bucket=bucket_for(score, threshold),
                meta=meta,
                details=output.details,
                trigger_frame_index=output.trigger_frame_index,
                frame_paths=frame_paths,
                model_scores=model_scores,
            )
        )
        if i % PROGRESS_EVERY == 0:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed else 0.0
            review = sum(1 for s in scored if s.bucket == "review")
            logger.info(
                "scored %d/%d (%.1f seq/s, %d review / %d unlabel)",
                i,
                total,
                rate,
                review,
                len(scored) - review,
            )
    return scored, dropped
