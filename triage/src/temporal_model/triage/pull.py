"""Pull the pyro-annotator unannotated backlog into the local store (read-only).

Incremental: a sequence already on disk is skipped, so recurring pulls only pay
for new sequences. meta.json is written last — a crashed frame download leaves
the sequence dir without a meta.json, so the next run re-pulls it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from temporal_model.triage.store import (
    IMAGES_DIR,
    FrameRef,
    SequenceMeta,
    sequence_dir,
    sequence_exists,
    write_meta,
)

logger = logging.getLogger(__name__)

DOWNLOAD_ATTEMPTS = 3
DEFAULT_WORKERS = 16  # per-sequence frame fetch+download concurrency
PROGRESS_EVERY = 25  # log a progress line every N pulled sequences


def _default_download(url: str) -> bytes:
    """Fetch one frame image over plain HTTP; retries absorb transient hiccups."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            logger.warning("frame download failed (attempt %d), retrying", attempt)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def pull_unannotated(
    client,
    store_dir: Path,
    *,
    processing_stage: str = "ready_to_annotate",
    limit: int | None = None,
    page_size: int = 100,
    workers: int = DEFAULT_WORKERS,
    download: Callable[[str], bytes] = _default_download,
) -> dict[str, int]:
    """Pull unannotated sequences + their frames. Returns {pulled, skipped}.

    Each sequence's frames are fetched + downloaded concurrently (``workers``
    threads); a progress line is logged every ``PROGRESS_EVERY`` pulled.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    pulled = skipped = 0
    for seq in client.iter_unannotated_sequences(
        processing_stage=processing_stage, page_size=page_size, limit=limit
    ):
        if sequence_exists(store_dir, seq["id"]):
            skipped += 1
            continue
        try:
            _pull_one(client, store_dir, seq, download, workers)
        except requests.RequestException:
            logger.exception(
                "pull failed for sequence %s; will retry next run", seq["id"]
            )
            continue
        pulled += 1
        if pulled % PROGRESS_EVERY == 0:
            elapsed = time.monotonic() - start
            rate = pulled / elapsed if elapsed else 0.0
            target = f"/{limit}" if limit else ""
            logger.info(
                "progress: %d%s pulled, %d skipped (%.1f seq/s)",
                pulled,
                target,
                skipped,
                rate,
            )
    logger.info("pull done: %d pulled, %d skipped", pulled, skipped)
    return {"pulled": pulled, "skipped": skipped}


def _pull_one(
    client, store_dir: Path, seq: dict, download: Callable[[str], bytes], workers: int
):
    dets = client.list_detections(seq["id"])
    dets = sorted(dets, key=lambda d: d["recorded_at"])
    meta = SequenceMeta(
        key=f"pyro-annotator_{seq['id']}",
        sequence_id=seq["id"],
        camera_id=seq.get("camera_id"),
        camera_name=seq.get("camera_name"),
        organization_id=seq.get("organisation_id"),
        organization_name=seq.get("organisation_name"),
        started_at=seq.get("recorded_at"),
        frames=[
            FrameRef(
                file=f"{IMAGES_DIR}/detection_{d['id']}.jpg",
                detection_id=d["id"],
                recorded_at=d.get("recorded_at"),
                bucket_key=d.get("bucket_key"),
            )
            for d in dets
        ],
    )
    seq_dir = sequence_dir(store_dir, meta)
    images_dir = seq_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    def fetch(det: dict) -> None:
        url = client.detection_image_url(det["id"])
        (images_dir / f"detection_{det['id']}.jpg").write_bytes(download(url))

    if workers <= 1 or len(dets) <= 1:
        for det in dets:
            fetch(det)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(dets))) as ex:
            # list() drains the iterator so the first download error propagates
            # (caught by pull_unannotated -> sequence re-pulled next run).
            list(ex.map(fetch, dets))
    # meta.json last: its presence marks the sequence complete.
    write_meta(seq_dir, meta)
