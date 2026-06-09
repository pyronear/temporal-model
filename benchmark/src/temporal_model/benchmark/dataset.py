"""Load a meta.json sequence store into core Frame objects.

Reads the same on-disk format the temporal-model-explorer writes
(``meta.json`` with an ordered ``frames`` list), but depends only on
``temporal_model.core`` — no explorer / pyrocore import.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from temporal_model.core.protocol import Frame

META_FILENAME = "meta.json"


@dataclass
class BenchSequence:
    """One benchmarkable sequence: its key, label, and ordered frames."""

    key: str
    label: str
    frame_count: int
    frames: list[Frame]


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build(seq_dir: Path, meta: dict) -> BenchSequence:
    frames = [
        Frame(
            frame_id=Path(ref["file"]).stem,
            image_path=seq_dir / ref["file"],
            timestamp=_parse_ts(ref.get("created_at")),
        )
        for ref in meta.get("frames", [])
    ]
    return BenchSequence(
        key=meta.get("key", seq_dir.name),
        label=meta.get("label", "unknown"),
        frame_count=len(frames),
        frames=frames,
    )


def iter_sequences(store_dir: Path) -> Iterator[BenchSequence]:
    """Yield one BenchSequence per meta.json under ``store_dir`` (recursive)."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        meta = json.loads(meta_path.read_text())
        yield _build(meta_path.parent, meta)
