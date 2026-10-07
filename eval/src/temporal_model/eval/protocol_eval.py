"""Protocol-level evaluation records + metrics for bbox-tube-temporal.

Operates on the output of ``BboxTubeTemporalModel.predict`` (the
pyrocore ``TemporalModel`` protocol) rather than on pre-built tube
patches.

Field names and rounding match the leaderboard's
``temporal_model_leaderboard.metrics.compute_metrics`` so numbers
produced here are directly comparable.
"""

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from temporal_model.core.protocol import Frame, TemporalModelOutput


@dataclass
class SequenceRecord:
    """One sequence's protocol-eval result.

    Attributes:
        sequence_id: Sequence directory name.
        label: ``"smoke"`` or ``"fp"``.
        is_positive: Model's binary decision.
        trigger_frame_index: Decision frame (0-based), or ``None`` if negative.
        score: Sequence-level score used for PR/ROC
            (``max(tube_logits)`` per the ``max_logit`` aggregation
            rule baked into the packaged config). ``-inf`` when no
            tubes survived filtering.
        num_tubes_kept: Tubes that passed the inference-time filter.
        tube_logits: Per-tube logits (in kept-tube order).
        ttd_frames: Time-to-detect in frames for TPs
            (= ``trigger_frame_index``), else ``None``.
        details: Passthrough of ``TemporalModelOutput.details``.
    """

    sequence_id: str
    label: str
    is_positive: bool
    trigger_frame_index: int | None
    score: float
    num_tubes_kept: int
    tube_logits: list[float]
    ttd_frames: int | None = None
    details: dict = field(default_factory=dict)


def _score_from_tube_logits(tube_logits: list[float]) -> float:
    """max(logits), or ``-inf`` for an empty tube list."""
    return max(tube_logits) if tube_logits else -math.inf


def build_record(
    *,
    sequence_dir: Path,
    label: str,
    frames: list[Frame],
    output: TemporalModelOutput,
) -> SequenceRecord:
    """Bundle a per-sequence eval record from the model's output + frames.

    ``frames`` is accepted for interface symmetry with the previous
    timestamp-based TTD computation but is no longer read — TTD is
    taken directly from ``output.trigger_frame_index`` per the pyrocore
    convention (frame timestamps in the pyro-dataset are unreliable).
    """
    kept = output.details.get("tubes", {}).get("kept", [])
    tube_logits = [float(t["logit"]) for t in kept]
    ground_truth = label == "smoke"
    ttd_frames = (
        output.trigger_frame_index
        if ground_truth
        and output.is_positive
        and output.trigger_frame_index is not None
        else None
    )
    return SequenceRecord(
        sequence_id=sequence_dir.name,
        label=label,
        is_positive=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        score=_score_from_tube_logits(tube_logits),
        num_tubes_kept=len(kept),
        tube_logits=tube_logits,
        ttd_frames=ttd_frames,
        details=dict(output.details),
    )


def wilson_ci95(successes: int, n: int) -> list[float] | None:
    """Wilson score 95% interval for a proportion; ``None`` when ``n == 0``."""
    if n == 0:
        return None
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(center - half, 0.0), 4), round(min(center + half, 1.0), 4)]


def fpr_at_recall(records: list[SequenceRecord], target: float = 0.95) -> float | None:
    """Lowest FPR over score thresholds whose recall reaches ``target``.

    A test-set threshold sweep on the sequence score, for ranking models —
    not the deployed operating point. Each finite score is a candidate
    threshold (``score >= t`` is positive, so tied scores move together, no
    interpolation); ``-inf`` records (no tube) are never positive. ``None``
    when no threshold reaches ``target`` or a class is missing.
    """
    pos = [r.score for r in records if r.label == "smoke"]
    neg = [r.score for r in records if r.label == "fp"]
    if not pos or not neg:
        return None
    best = None
    for t in {s for s in pos if math.isfinite(s)}:
        if sum(s >= t for s in pos) / len(pos) >= target:
            fpr = sum(s >= t for s in neg) / len(neg)
            best = fpr if best is None else min(best, fpr)
    return None if best is None else round(best, 4)


def detected_within_frames(
    records: list[SequenceRecord], ks: tuple[int, ...] = (2, 3, 5)
) -> dict[str, float | None]:
    """Share of all smoke sequences alerted within their first k frames.

    ``ttd_frames`` is zero-based, so "within k" is ``ttd_frames < k``; the
    earliest possible alert is the 2nd frame (``infer_min_tube_length: 2``),
    hence k starts at 2. Missed
    smoke counts as not detected, unlike the TTD mean/median, which only see
    detected smoke and so reward a model for missing the hard ones.
    """
    ttds = [r.ttd_frames for r in records if r.label == "smoke"]
    return {
        str(k): (
            round(sum(t is not None and t < k for t in ttds) / len(ttds), 4)
            if ttds
            else None
        )
        for k in ks
    }


def compute_metrics(model_name: str, records: list[SequenceRecord]) -> dict:
    """Aggregate leaderboard-style metrics + PR/ROC AUCs over records.

    Returns a plain dict so it serializes with ``json.dumps`` directly.
    Field names / rounding match
    ``temporal_model_leaderboard.metrics.ModelMetrics``.
    """
    tp = sum(1 for r in records if r.label == "smoke" and r.is_positive)
    fp = sum(1 for r in records if r.label == "fp" and r.is_positive)
    fn = sum(1 for r in records if r.label == "smoke" and not r.is_positive)
    tn = sum(1 for r in records if r.label == "fp" and not r.is_positive)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    n_neg = fp + tn
    fpr = fp / n_neg if n_neg > 0 else 0.0

    ttd_values = [
        r.ttd_frames
        for r in records
        if r.label == "smoke" and r.is_positive and r.ttd_frames is not None
    ]
    mean_ttd = round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    median_ttd = round(statistics.median(ttd_values), 1) if ttd_values else None

    y_true = np.array([1 if r.label == "smoke" else 0 for r in records])
    scores = np.array([r.score for r in records], dtype=float)
    # sklearn rejects ±inf — clip to finite range before AUC computation.
    scores_finite = np.clip(scores, np.finfo(float).min, np.finfo(float).max)
    pr_auc = (
        float(average_precision_score(y_true, scores_finite))
        if y_true.sum() > 0
        else 0.0
    )
    roc_auc = (
        float(roc_auc_score(y_true, scores_finite))
        if 0 < y_true.sum() < len(y_true)
        else 0.0
    )

    return {
        "model_name": model_name,
        "num_sequences": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mean_ttd_frames": mean_ttd,
        "median_ttd_frames": median_ttd,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_ci95": wilson_ci95(tp, tp + fn),
        "fpr_ci95": wilson_ci95(fp, n_neg),
        "fpr_at_recall_95": fpr_at_recall(records, 0.95),
        "detected_within_frames": detected_within_frames(records),
    }
