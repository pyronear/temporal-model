"""Import scored sequences (frames + provenance) from alert-api into the store.

Incremental by design: a sequence already in the store is skipped, so a
recurring import only pays for new sequences. Detections arrive oldest first
and are stored one FrameMeta per detection (a bucket_key can repeat when
several detections share a frame; replay deduplicates, mirroring production).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from pathlib import Path

import requests

from temporal_model.monitor.store import (
    IMAGES_DIR,
    FrameMeta,
    SequenceMeta,
    label_from_is_wildfire,
    sequence_dir,
    sequence_exists,
    write_meta,
)

logger = logging.getLogger(__name__)


def _default_download(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def _date_range(day_from: str, day_to: str) -> list[str]:
    start = dt.date.fromisoformat(day_from)
    end = dt.date.fromisoformat(day_to)
    return [
        (start + dt.timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]


def _camera_index(client) -> dict[int, dict]:
    return {cam["id"]: cam for cam in client.list_cameras()}


def _org_names(client) -> dict[int, str]:
    try:
        return {org["id"]: org["name"] for org in client.list_organizations()}
    except Exception:  # noqa: BLE001 — org listing may need admin scope
        logger.warning("organizations endpoint unavailable; using org-<id> names")
        return {}


def import_platform(
    client,
    store_dir: Path,
    day_from: str,
    day_to: str,
    *,
    force: bool = False,
    download: Callable[[str], bytes] = _default_download,
) -> dict[str, int]:
    """Import all sequences in [day_from, day_to] (inclusive). Returns counts."""
    store_dir.mkdir(parents=True, exist_ok=True)
    cameras = _camera_index(client)
    org_names = _org_names(client)
    imported = skipped = 0
    for day in _date_range(day_from, day_to):
        for seq in client.list_sequences_for_date(day):
            if not force and sequence_exists(store_dir, seq["id"]):
                skipped += 1
                continue
            _import_one(client, store_dir, seq, cameras, org_names, download)
            imported += 1
    logger.info("import done: %d imported, %d skipped", imported, skipped)
    return {"imported": imported, "skipped": skipped}


def _import_one(
    client,
    store_dir: Path,
    seq: dict,
    cameras: dict[int, dict],
    org_names: dict[int, str],
    download: Callable[[str], bytes],
) -> None:
    dets = client.list_sequence_detections(seq["id"])
    dets = sorted(dets, key=lambda d: d["created_at"])
    camera = cameras.get(seq.get("camera_id")) or {}
    org_id = camera.get("organization_id")
    label, label_detail = label_from_is_wildfire(seq.get("is_wildfire"))
    meta = SequenceMeta(
        key=f"platform_{seq['id']}",
        sequence_id=seq["id"],
        label=label,
        label_detail=label_detail,
        camera_id=seq.get("camera_id"),
        camera_name=camera.get("name"),
        organization_id=org_id,
        organization_name=org_names.get(org_id)
        or (f"org-{org_id}" if org_id is not None else None),
        started_at=seq.get("started_at"),
        temporal_model_score=seq.get("temporal_model_score"),
        temporal_model_version=seq.get("temporal_model_version"),
        temporal_api_version=seq.get("temporal_api_version"),
        frames=[
            FrameMeta(
                file=f"{IMAGES_DIR}/detection_{d['id']}.jpg",
                detection_id=d["id"],
                created_at=d["created_at"],
                bucket_key=d["bucket_key"],
                bbox=d.get("bbox") or "",
            )
            for d in dets
        ],
    )
    seq_dir = sequence_dir(store_dir, meta)
    images_dir = seq_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    for det in dets:
        (images_dir / f"detection_{det['id']}.jpg").write_bytes(download(det["url"]))
    # meta.json last: its presence marks the sequence complete, so a crashed
    # download is re-fetched (not skipped) on the next run.
    write_meta(seq_dir, meta)
