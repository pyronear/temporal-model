"""Tests for benchmark aggregation."""

import pandas as pd

from temporal_model.benchmark.report import summarize

STAGES = ["pad", "yolo", "tubes", "crop", "vit", "trigger"]


def _row(key, total, **stage_ms):
    r = {
        "key": key,
        "rep": 0,
        "failed": False,
        "frame_count": 6,
        "n_kept_tubes": 1,
        "total_ms": total,
    }
    for s in STAGES:
        r[f"{s}_ms"] = stage_ms.get(s, 0.0)
    return r


def test_summarize_latency_percentiles_and_counts():
    df = pd.DataFrame(
        [
            _row("a", 100.0, vit=80.0, yolo=20.0),
            _row("b", 200.0, vit=160.0, yolo=40.0),
            _row("c", 300.0, vit=240.0, yolo=60.0),
        ]
    )
    s = summarize(df)
    assert s["n_sequences"] == 3
    assert s["n_failed"] == 0
    assert s["total_ms"]["p50"] == 200.0
    # frames/sec uses mean latency over total frames; just assert it's positive.
    assert s["throughput"]["sequences_per_sec"] > 0
    # vit dominates the mean stage share.
    assert s["stage_share_pct"]["vit"] > s["stage_share_pct"]["yolo"]


def test_summarize_counts_failures_and_excludes_them():
    df = pd.DataFrame(
        [
            _row("a", 100.0, vit=100.0),
            {"key": "b", "rep": 0, "failed": True},
        ]
    )
    s = summarize(df)
    assert s["n_sequences"] == 2
    assert s["n_failed"] == 1
    assert s["total_ms"]["p50"] == 100.0  # failed row excluded from latency
