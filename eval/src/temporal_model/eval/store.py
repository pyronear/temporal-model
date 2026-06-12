"""Common local sequence store: types, meta.json IO, label + Frame helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from temporal_model.core.protocol import Frame

META_FILENAME = "meta.json"


def slug(value: str) -> str:
    """On-disk-safe slug: lowercased, spaces and slashes become dashes."""
    return value.strip().lower().replace(" ", "-").replace("/", "-")


@dataclass
class FrameRef:
    file: str  # path relative to the sequence dir, e.g. "images/detection_5.jpg"
    detection_id: int | None = None
    created_at: str | None = None  # ISO timestamp (platform) or None (zip)


@dataclass
class SequenceMeta:
    key: str
    sequence_id: str
    source: str  # "platform"
    label: str  # "smoke" | "fp" | "unknown"
    label_detail: str | None
    label_source: str  # "platform_is_wildfire"
    frames: list[FrameRef] = field(default_factory=list)
    camera_id: int | None = None
    camera_name: str | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    started_at: str | None = None


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_FILENAME).write_text(json.dumps(asdict(meta), indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    payload = json.loads((seq_dir / META_FILENAME).read_text())
    frames = [FrameRef(**f) for f in payload.pop("frames", [])]
    return SequenceMeta(frames=frames, **payload)


def iter_sequence_dirs(store_dir: Path) -> Iterator[Path]:
    """Yield every directory under ``store_dir`` containing a meta.json (recursive)."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        yield meta_path.parent


def normalize_label(
    raw: str | None, smoke_values: list[str], fp_values: list[str]
) -> str:
    """Normalize a raw category to the tri-state keep/discard label."""
    if not raw:
        return "unknown"
    v = raw.lower()
    if v in {s.lower() for s in smoke_values} or "smoke" in v or v == "wildfire":
        return "smoke"
    if v in {f.lower() for f in fp_values}:
        return "fp"
    return "unknown"


def build_frames(seq_dir: Path, meta: SequenceMeta) -> list[Frame]:
    """Build the ordered pyrocore Frame list the model consumes.

    Meta order is the time axis.
    """
    frames: list[Frame] = []
    for ref in meta.frames:
        ts = None
        if ref.created_at:
            try:
                ts = datetime.fromisoformat(ref.created_at.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        frames.append(
            Frame(
                frame_id=Path(ref.file).stem,
                image_path=seq_dir / ref.file,
                timestamp=ts,
            )
        )
    return frames
