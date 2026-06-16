"""Tar-shard the triage working tree so DVC tracks ~tens of objects, not ~300k.

At full scale the store is ~247k frame files and the report ~43k per-key JSON
files — a DVC object explosion. ``pack`` bundles per-sequence data into sealed
tar shards (frames + meta, and details + sequence-view), leaving only the
handful of aggregate report files loose; ``unpack`` restores the loose working
tree so ``score`` and the viewer read files by path, unchanged.

Two shard sets with different lifecycles:
- ``frames/`` — append-only and model-independent (a sequence's images never
  change); a re-score never rewrites them.
- ``report/`` — rebuilt each run (predictions are per model run).

Each shard is sealed at ``target_bytes`` (~1 GB); tars are uncompressed (JPEGs
already are). A ``manifest.json`` per set maps unit id → shard for incremental
packs and tooling.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

from temporal_model.triage.store import iter_sequence_dirs, read_meta

FRAMES_DIR = "frames"
REPORT_DIR = "report"
MANIFEST = "manifest.json"
DEFAULT_TARGET_BYTES = 1_000_000_000  # ~1 GB per shard

# Few, small, aggregate report files kept loose alongside the shards (and pushed).
AGGREGATE_FILES = [
    "results.json",
    "results.parquet",
    "model_config.json",
    "dropped.json",
    "unlabeled.json",
    "review.json",
]


def _read_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"next_index": 1, "items": {}}


def _seal_units(
    units: list,
    *,
    unit_id: Callable[[object], str],
    unit_size: Callable[[object], int],
    add_to_tar: Callable[[tarfile.TarFile, object], None],
    shard_dir: Path,
    manifest: dict,
    target_bytes: int,
) -> None:
    """Append ``units`` into new sealed tars, updating ``manifest`` in place.

    A unit is never split across shards; a shard is sealed once adding the next
    unit would exceed ``target_bytes`` (a single oversized unit still gets its
    own shard).
    """
    if not units:
        return
    idx: int = manifest["next_index"]
    tar: tarfile.TarFile | None = None
    shard_name = ""
    used = 0
    for u in units:
        size = unit_size(u)
        if tar is None or (used > 0 and used + size > target_bytes):
            if tar is not None:
                tar.close()
                idx += 1
            shard_name = f"shard_{idx:04d}.tar"
            # Lifecycle spans loop iterations (sealing on size), so no `with`.
            tar = tarfile.open(shard_dir / shard_name, "w")  # noqa: SIM115
            used = 0
        add_to_tar(tar, u)
        manifest["items"][unit_id(u)] = shard_name
        used += size
    if tar is not None:
        tar.close()
        idx += 1
    manifest["next_index"] = idx


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def pack_frames(store_dir: Path, shards_dir: Path, *, target_bytes: int) -> dict:
    """Append-only: pack store sequences absent from the manifest into new tars."""
    fdir = shards_dir / FRAMES_DIR
    fdir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(fdir / MANIFEST)
    seq_dirs = [
        d
        for d in iter_sequence_dirs(store_dir)
        if str(read_meta(d).sequence_id) not in manifest["items"]
    ]
    _seal_units(
        seq_dirs,
        unit_id=lambda d: str(read_meta(d).sequence_id),
        unit_size=_dir_size,
        add_to_tar=lambda tar, d: tar.add(
            d, arcname=d.relative_to(store_dir).as_posix()
        ),
        shard_dir=fdir,
        manifest=manifest,
        target_bytes=target_bytes,
    )
    (fdir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def pack_report(report_dir: Path, shards_dir: Path, *, target_bytes: int) -> dict:
    """Rebuild the report shards from scratch (predictions are per run)."""
    rdir = shards_dir / REPORT_DIR
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.stem for p in (report_dir / "details").glob("*.json"))
    manifest = {"next_index": 1, "items": {}}

    def members(key: str) -> list[tuple[Path, str]]:
        return [
            (report_dir / "details" / f"{key}.json", f"details/{key}.json"),
            (report_dir / "sequences" / f"{key}.json", f"sequences/{key}.json"),
        ]

    _seal_units(
        keys,
        unit_id=lambda k: k,
        unit_size=lambda k: sum(p.stat().st_size for p, _ in members(k) if p.exists()),
        add_to_tar=lambda tar, k: [
            tar.add(p, arcname=arc) for p, arc in members(k) if p.exists()
        ],
        shard_dir=rdir,
        manifest=manifest,
        target_bytes=target_bytes,
    )
    (rdir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def pack(
    store_dir: Path,
    report_dir: Path,
    shards_dir: Path,
    *,
    target_bytes: int = DEFAULT_TARGET_BYTES,
) -> dict[str, int]:
    """Pack the working tree into ``shards_dir``. Returns shard counts."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    frames = pack_frames(store_dir, shards_dir, target_bytes=target_bytes)
    report = pack_report(report_dir, shards_dir, target_bytes=target_bytes)
    for name in AGGREGATE_FILES:
        src = report_dir / name
        if src.exists():
            shutil.copy2(src, shards_dir / name)
    return {
        "frame_shards": frames["next_index"] - 1,
        "report_shards": report["next_index"] - 1,
        "sequences": len(frames["items"]),
    }


def unpack(shards_dir: Path, store_dir: Path, report_dir: Path) -> None:
    """Restore the loose store + report from ``shards_dir`` (idempotent extract)."""
    store_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for tar_path in sorted((shards_dir / FRAMES_DIR).glob("shard_*.tar")):
        with tarfile.open(tar_path) as tar:
            tar.extractall(store_dir, filter="data")
    for tar_path in sorted((shards_dir / REPORT_DIR).glob("shard_*.tar")):
        with tarfile.open(tar_path) as tar:
            tar.extractall(report_dir, filter="data")
    for name in AGGREGATE_FILES:
        src = shards_dir / name
        if src.exists():
            shutil.copy2(src, report_dir / name)
