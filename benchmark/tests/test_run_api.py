"""Tests for the API benchmark client's request planning + row assembly."""

from temporal_model.benchmark.run_api import (
    _http_post,
    build_requests,
    frame_key,
    rows_for_sequence,
)
from temporal_model.core.protocol import Frame


class _FakeResp:
    status_code = 200

    def json(self):
        return {}


def test_http_post_sends_bearer_token_when_env_set(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.setenv("TEMPORAL_API_TOKEN", "s3cr3t")
    monkeypatch.setattr("temporal_model.benchmark.run_api.requests.post", fake_post)
    _http_post("http://x", ["a.jpg"])
    assert captured["headers"] == {"Authorization": "Bearer s3cr3t"}


def test_http_post_no_auth_header_when_env_unset(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _FakeResp()

    monkeypatch.delenv("TEMPORAL_API_TOKEN", raising=False)
    monkeypatch.setattr("temporal_model.benchmark.run_api.requests.post", fake_post)
    _http_post("http://x", ["a.jpg"])
    assert captured["headers"] is None


def _seq(store, n):
    return [
        Frame(frame_id=f"f{i}", image_path=store / "org/cam/seq" / f"{i}.jpg")
        for i in range(n)
    ]


def test_frame_key_is_store_relative_posix(tmp_path):
    f = Frame(frame_id="x", image_path=tmp_path / "a/b/c.jpg")
    assert frame_key(tmp_path, f) == "a/b/c.jpg"


def test_build_requests_cold_is_single_full_list(tmp_path):
    frames = _seq(tmp_path, 5)
    reqs = build_requests(tmp_path, frames, "cold", warm_min_frames=3)
    assert len(reqs) == 1
    prefix_len, keys = reqs[0]
    assert prefix_len == 5
    assert len(keys) == 5


def test_build_requests_warm_is_growing_prefixes(tmp_path):
    frames = _seq(tmp_path, 5)
    reqs = build_requests(tmp_path, frames, "warm", warm_min_frames=3)
    assert [p for p, _ in reqs] == [3, 4, 5]
    assert [len(k) for _, k in reqs] == [3, 4, 5]


def test_build_requests_warm_short_sequence(tmp_path):
    frames = _seq(tmp_path, 2)  # shorter than warm_min_frames
    reqs = build_requests(tmp_path, frames, "warm", warm_min_frames=3)
    assert [p for p, _ in reqs] == [2]


def test_rows_for_sequence_flattens_profiling(tmp_path):
    frames = _seq(tmp_path, 3)

    def fake_post(url, keys):
        body = {
            "details": {
                "profiling": {
                    "stages_ms": {
                        "s3_fetch": 5.0,
                        "detector": 50.0,
                        "classifier": 10.0,
                    },
                    "total_ms": 65.0,
                    "n_frames": len(keys),
                    "cache_hits": 0,
                    "cache_misses": len(keys),
                }
            }
        }
        return 200, body, 70.0  # status, json, e2e_ms

    rows = rows_for_sequence(
        tmp_path,
        "k1",
        frames,
        "cold",
        warm_min_frames=3,
        base_url="http://x",
        post=fake_post,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["pass"] == "cold"
    assert row["key"] == "k1"
    assert row["prefix_len"] == 3
    assert row["http_status"] == 200
    assert row["e2e_ms"] == 70.0
    assert row["detector_ms"] == 50.0
    assert row["total_ms"] == 65.0
    assert row["cache_misses"] == 3
