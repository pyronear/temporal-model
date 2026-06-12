"""Import scored sequences (frames + provenance) from alert-api into the store.

Incremental by design: a sequence already in the store is skipped, so a
recurring import only pays for new sequences. Detections arrive oldest first
and are stored one FrameMeta per detection (a bucket_key can repeat when
several detections share a frame; replay deduplicates, mirroring production).
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import requests

from temporal_model.monitor.store import (
    IMAGES_DIR,
    FrameMeta,
    SequenceMeta,
    find_sequence_dirs,
    iter_metas,
    label_from_is_wildfire,
    sequence_dir,
    sequence_exists,
    slugify,
    write_meta,
)

logger = logging.getLogger(__name__)


DOWNLOAD_ATTEMPTS = 3


def _default_download(url: str) -> bytes:
    """Fetch one frame; retries absorb transient S3/CDN hiccups."""
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


def _org_name(
    seq: dict, cameras: dict[int, dict], org_names: dict[int, str]
) -> str | None:
    """Resolve the org name for a sequence (same logic used by _import_one)."""
    camera = cameras.get(seq.get("camera_id")) or {}
    org_id = camera.get("organization_id")
    return org_names.get(org_id) or (f"org-{org_id}" if org_id is not None else None)


def import_alert_api(
    client,
    store_dir: Path,
    day_from: str,
    day_to: str,
    *,
    force: bool = False,
    download: Callable[[str], bytes] = _default_download,
    exclude_orgs: set[str] | None = None,
) -> dict[str, int]:
    """Import all sequences in [day_from, day_to] (inclusive). Returns counts."""
    store_dir.mkdir(parents=True, exist_ok=True)
    cameras = _camera_index(client)
    org_names = _org_names(client)
    imported = skipped = excluded = 0
    for day in _date_range(day_from, day_to):
        for seq in client.list_sequences_for_date(day):
            org_slug = slugify(_org_name(seq, cameras, org_names))
            if exclude_orgs and org_slug in exclude_orgs:
                excluded += 1
                continue
            if not force and sequence_exists(store_dir, seq["id"]):
                skipped += 1
                continue
            try:
                _import_one(client, store_dir, seq, cameras, org_names, download)
            except requests.RequestException:
                logger.exception(
                    "import failed for sequence %s; will retry next run", seq["id"]
                )
                continue
            imported += 1
    logger.info(
        "import done: %d imported, %d skipped, %d excluded", imported, skipped, excluded
    )
    return {"imported": imported, "skipped": skipped, "excluded": excluded}


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
        key=f"alert-api_{seq['id']}",
        sequence_id=seq["id"],
        label=label,
        label_detail=label_detail,
        camera_id=seq.get("camera_id"),
        camera_name=camera.get("name"),
        organization_id=org_id,
        organization_name=_org_name(seq, cameras, org_names),
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
    # Remove stale dirs from a previous import under a different org/camera name.
    for stale in find_sequence_dirs(store_dir, seq["id"]):
        if stale != seq_dir:
            shutil.rmtree(stale)
    images_dir = seq_dir / IMAGES_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    for det in dets:
        (images_dir / f"detection_{det['id']}.jpg").write_bytes(download(det["url"]))
    # meta.json last: its presence marks the sequence complete, so a crashed
    # download is re-fetched (not skipped) on the next run.
    write_meta(seq_dir, meta)


HEAD_GAP = 25  # consecutive missing IDs that mean "past the newest sequence"
OLDER_STOP = 25  # consecutive too-old sequences that mean "past the date range"
# Mid-walk 404 holes (deleted sequences) are a different concern from head
# detection: a bulk deletion can leave a far wider gap than HEAD_GAP without
# meaning the scan is done — only the date rule ends the downward walk.
WALK_GAP = 500


def import_all_orgs(
    client,
    store_dir: Path,
    day_from: str,
    day_to: str,
    *,
    force: bool = False,
    seed_id: int | None = None,
    download: Callable[[str], bytes] = _default_download,
    exclude_orgs: set[str] | None = None,
) -> dict[str, int]:
    """Import sequences of EVERY organization by scanning the global ID space.

    The listing endpoint is locked to the token's organization, but an admin
    token can read any sequence by id and ids are a global autoincrement,
    roughly monotone with started_at. Strategy: seed from the store's max id
    (or the own-org listing, or --seed-id), probe upward until HEAD_GAP
    consecutive 404s (the head), then walk downward importing sequences whose
    started_at date falls within [day_from, day_to], stopping after
    OLDER_STOP consecutive sequences older than day_from. 404 holes (deleted
    sequences) are skipped in both directions.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    seed = (
        seed_id or _max_store_id(store_dir) or _max_listed_id(client, day_from, day_to)
    )
    if seed is None:
        raise SystemExit(
            "cannot seed the id scan: store is empty and the account's own-org "
            "listing returned nothing — pass --seed-id <recent sequence id>"
        )
    head = _find_head(client, seed)
    cameras = _camera_index(client)
    org_names = _org_names(client)
    imported = skipped = excluded = 0
    misses = older = 0
    sid = head
    while sid > 0 and misses < WALK_GAP and older < OLDER_STOP:
        seq = client.get_sequence(sid)
        sid -= 1
        if seq is None:
            misses += 1
            continue
        misses = 0
        day = (seq.get("started_at") or "")[:10]
        if day > day_to:
            # ids are only roughly monotone with started_at: a newer-than-range
            # sequence here breaks the "consecutive older" streak too
            older = 0
            continue
        if day < day_from:
            older += 1
            continue
        older = 0
        if exclude_orgs and slugify(_org_name(seq, cameras, org_names)) in exclude_orgs:
            excluded += 1
            continue
        if not force and sequence_exists(store_dir, seq["id"]):
            skipped += 1
            continue
        try:
            _import_one(client, store_dir, seq, cameras, org_names, download)
        except requests.RequestException:
            logger.exception(
                "import failed for sequence %s; will retry next run", seq["id"]
            )
            continue
        imported += 1
    logger.info(
        "scan import done: %d imported, %d skipped, %d excluded",
        imported,
        skipped,
        excluded,
    )
    return {"imported": imported, "skipped": skipped, "excluded": excluded}


def _max_store_id(store_dir: Path) -> int | None:
    ids = [meta.sequence_id for _, meta in iter_metas(store_dir)]
    return max(ids) if ids else None


def _max_listed_id(client, day_from: str, day_to: str) -> int | None:
    ids = [
        s["id"]
        for day in _date_range(day_from, day_to)
        for s in client.list_sequences_for_date(day)
    ]
    return max(ids) if ids else None


def _find_head(client, seed: int) -> int:
    """Highest existing sequence id at or above ``seed``."""
    head = seed
    misses = 0
    sid = seed + 1
    while misses < HEAD_GAP:
        if client.get_sequence(sid) is None:
            misses += 1
        else:
            head = sid
            misses = 0
        sid += 1
    return head
