"""Pure decision/outcome helpers + results filtering (no I/O, no Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

ERROR_OUTCOMES = {"discarded-smoke", "kept-fp"}


def decision_from_output(is_positive: bool) -> str:
    """Map a model's is_positive verdict to keep/discard."""
    return "keep" if is_positive else "discard"


def max_probability(details: dict[str, Any] | None) -> float | None:
    """Largest calibrated probability across kept tubes, or None."""
    kept = (details or {}).get("tubes", {}).get("kept", [])
    probs = [t.get("probability") for t in kept if t.get("probability") is not None]
    return max(probs) if probs else None


def compute_outcome(decision: str, label: str) -> str:
    """Outcome of a decision vs the ground-truth label."""
    if label == "smoke":
        return "kept-smoke" if decision == "keep" else "discarded-smoke"
    if label == "fp":
        return "kept-fp" if decision == "keep" else "discarded-fp"
    return "n/a"


def filter_results(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    decision: str | None = None,
    label: str | None = None,
    outcome: str | None = None,
    source: str | None = None,
    camera_name: str | None = None,
    organization_name: str | None = None,
    errors_only: bool = False,
) -> pd.DataFrame:
    """Filter a results DataFrame by any combination of column values."""
    out = df
    for col, val in (
        ("model", model),
        ("decision", decision),
        ("label", label),
        ("outcome", outcome),
        ("source", source),
        ("camera_name", camera_name),
        ("organization_name", organization_name),
    ):
        if val is not None:
            out = out[out[col] == val]
    if errors_only:
        out = out[out["outcome"].isin(ERROR_OUTCOMES)]
    return out


def performance_summary(df: pd.DataFrame) -> dict:
    """Headline metrics over labeled rows (label in {smoke, fp}) of ``df``.

    ``df`` is expected to already be narrowed to one source + model. Returns
    counts plus recall / specificity (FP-filtered) / precision, each ``None``
    when its denominator is 0. Counts are derived from the ``outcome`` column.
    """
    oc = df[df["label"].isin(("smoke", "fp"))]["outcome"]
    kept_smoke = int((oc == "kept-smoke").sum())
    discarded_smoke = int((oc == "discarded-smoke").sum())
    discarded_fp = int((oc == "discarded-fp").sum())
    kept_fp = int((oc == "kept-fp").sum())
    n_smoke = kept_smoke + discarded_smoke
    n_fp = discarded_fp + kept_fp
    n_kept = kept_smoke + kept_fp
    return {
        "n_labeled": n_smoke + n_fp,
        "n_smoke": n_smoke,
        "n_fp": n_fp,
        "kept_smoke": kept_smoke,
        "discarded_smoke": discarded_smoke,
        "discarded_fp": discarded_fp,
        "kept_fp": kept_fp,
        "recall": kept_smoke / n_smoke if n_smoke else None,
        "specificity": discarded_fp / n_fp if n_fp else None,
        "precision": kept_smoke / n_kept if n_kept else None,
    }


def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Re-decide keep/discard at ``threshold`` from the ``probability`` column.

    Returns a copy: ``decision`` = keep iff ``probability`` is non-null and
    ``>= threshold`` (a sequence with no kept tubes has null probability ->
    discard), and ``outcome`` recomputed via :func:`compute_outcome`. All other
    columns (``score``, ``probability``, ``label``, metadata) are unchanged.
    """
    out = df.copy()
    keep = out["probability"].notna() & (out["probability"] >= threshold)
    out["decision"] = keep.map({True: "keep", False: "discard"})
    out["outcome"] = [
        compute_outcome(d, lbl)
        for d, lbl in zip(out["decision"], out["label"], strict=True)
    ]
    return out
