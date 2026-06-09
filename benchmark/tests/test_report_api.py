"""Tests for API benchmark aggregation."""

import json

import pandas as pd

from temporal_model.benchmark.report import summarize_api


def _row(pass_name, e2e, detector, status=200, hits=0, misses=6):
    return {
        "pass": pass_name,
        "key": "k",
        "prefix_len": 6,
        "e2e_ms": e2e,
        "http_status": status,
        "s3_fetch_ms": 5.0,
        "detector_ms": detector,
        "classifier_ms": 10.0,
        "total_ms": detector + 15.0,
        "n_frames": 6,
        "cache_hits": hits,
        "cache_misses": misses,
    }


def test_summarize_api_splits_cold_and_warm():
    df = pd.DataFrame(
        [
            _row("cold", 100.0, 60.0, hits=0, misses=6),
            _row("cold", 200.0, 120.0, hits=0, misses=6),
            _row("warm", 30.0, 5.0, hits=5, misses=1),
            _row("warm", 50.0, 7.0, hits=6, misses=0),
        ]
    )
    s = summarize_api(df)
    assert set(s["passes"]) == {"cold", "warm"}
    assert s["passes"]["cold"]["e2e_ms"]["p50"] == 150.0  # quantile(.5) of [100,200]
    assert s["passes"]["cold"]["n_requests"] == 2
    # warm amortizes the detector vs cold
    assert (
        s["passes"]["warm"]["stage_ms_mean"]["detector"]
        < s["passes"]["cold"]["stage_ms_mean"]["detector"]
    )
    # warm cache hit rate = 11 hits / (11 hits + 1 miss)
    assert round(s["passes"]["warm"]["cache_hit_rate"], 3) == round(11 / 12, 3)
    # JSON-serializable
    assert json.loads(json.dumps(s)) == s


def test_summarize_api_counts_errors():
    df = pd.DataFrame(
        [
            _row("cold", 100.0, 60.0),
            {"pass": "cold", "key": "k2", "http_status": 0, "e2e_ms": 0.0},
        ]
    )
    s = summarize_api(df)
    assert s["passes"]["cold"]["n_errors"] == 1
    assert s["passes"]["cold"]["n_requests"] == 2


def test_summarize_api_empty_frame_is_safe():
    s = summarize_api(pd.DataFrame())
    assert s == {"passes": {}, "n_requests": 0}
