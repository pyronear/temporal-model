"""Drive the API over HTTP in cold and warm cache passes, collecting per-request
e2e latency and the server-side profiling block."""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from temporal_model.core.protocol import Frame

from .dataset import iter_sequences

logger = logging.getLogger(__name__)


def frame_key(store_dir: Path, frame: Frame) -> str:
    """S3 key for a frame: its path relative to the store root (POSIX)."""
    return frame.image_path.relative_to(store_dir).as_posix()


def build_requests(
    store_dir: Path,
    frames: list[Frame],
    pass_name: str,
    *,
    warm_min_frames: int,
) -> list[tuple[int, list[str]]]:
    """Plan the requests for one sequence under a cache pass.

    cold → one request with the full key list. warm → growing prefixes
    ``warm_min_frames .. N`` (clamped to N for short sequences).
    """
    keys = [frame_key(store_dir, f) for f in frames]
    n = len(keys)
    if pass_name == "cold":
        return [(n, keys)]
    start = min(warm_min_frames, n)
    return [(k, keys[:k]) for k in range(start, n + 1)]


def _http_post(base_url: str, keys: list[str]) -> tuple[int, dict, float]:
    """POST one request; return (status, json_body, e2e_ms)."""
    token = os.environ.get("TEMPORAL_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    started = time.perf_counter()
    resp = requests.post(
        f"{base_url}/predict?verbose=true",
        json={"frames": keys},
        headers=headers,
        timeout=300,
    )
    e2e_ms = (time.perf_counter() - started) * 1000.0
    try:
        body = resp.json()
    except ValueError:
        body = {}
    return resp.status_code, body, e2e_ms


def rows_for_sequence(
    store_dir: Path,
    key: str,
    frames: list[Frame],
    pass_name: str,
    *,
    warm_min_frames: int,
    base_url: str,
    post=_http_post,
) -> list[dict]:
    """Issue all requests for one sequence and return one raw row per request."""
    rows: list[dict] = []
    for prefix_len, keys in build_requests(
        store_dir, frames, pass_name, warm_min_frames=warm_min_frames
    ):
        status, body, e2e_ms = post(base_url, keys)
        row = {
            "pass": pass_name,
            "key": key,
            "prefix_len": prefix_len,
            "e2e_ms": round(e2e_ms, 3),
            "http_status": status,
        }
        prof = (body.get("details") or {}).get("profiling") if body else None
        if prof:
            for stage, ms in prof.get("stages_ms", {}).items():
                row[f"{stage}_ms"] = ms
            row["total_ms"] = prof.get("total_ms")
            row["n_frames"] = prof.get("n_frames")
            row["cache_hits"] = prof.get("cache_hits")
            row["cache_misses"] = prof.get("cache_misses")
        rows.append(row)
    return rows


def run_api(
    store_dir: Path,
    base_url: str,
    *,
    passes: tuple[str, ...] = ("cold", "warm"),
    warmup: int = 3,
    limit: int | None = None,
    warm_min_frames: int = 3,
) -> pd.DataFrame:
    """Benchmark the API over HTTP; one row per request, per pass."""
    all_sequences = list(iter_sequences(store_dir))
    if limit is not None:
        all_sequences = all_sequences[: limit + warmup]
    if not all_sequences:
        raise SystemExit(f"no sequences found under {store_dir}")

    # Warmup sequences populate the server detection cache, so they must NOT be
    # measured — otherwise the cold pass would re-hit a warm cache for them.
    warmup_seqs = all_sequences[:warmup]
    sequences = all_sequences[warmup:]
    for seq in warmup_seqs:
        rows_for_sequence(
            store_dir,
            seq.key,
            seq.frames,
            "cold",
            warm_min_frames=warm_min_frames,
            base_url=base_url,
        )

    rows: list[dict] = []
    for pass_name in passes:
        for i, seq in enumerate(sequences):
            try:
                rows.extend(
                    rows_for_sequence(
                        store_dir,
                        seq.key,
                        seq.frames,
                        pass_name,
                        warm_min_frames=warm_min_frames,
                        base_url=base_url,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — record + continue
                logger.warning("sequence %s (%s) failed: %s", seq.key, pass_name, exc)
                rows.append(
                    {
                        "pass": pass_name,
                        "key": seq.key,
                        "http_status": 0,
                        "e2e_ms": 0.0,
                    }
                )
            if (i + 1) % 25 == 0:
                logger.info("%s: %d/%d sequences", pass_name, i + 1, len(sequences))

    return pd.DataFrame(rows)
