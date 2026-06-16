"""Sequence store: data/01_raw/sequences/<org>/<camera>/seq_<id>/{meta.json, images/}.

Holds the pyro-annotator unannotated backlog. These sequences carry no human
label yet, so ``label`` is always ``"unknown"``; per-frame provenance keeps the
annotator ``detection_id``, capture time, and S3 ``bucket_key``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from temporal_model.core.protocol import Frame

META_NAME = "meta.json"
IMAGES_DIR = "images"


class FrameRef(BaseModel):
    file: str  # relative to the sequence dir, e.g. "images/detection_7.jpg"
    detection_id: int
    recorded_at: str | None = None
    bucket_key: str | None = None


class SequenceMeta(BaseModel):
    key: str  # "pyro-annotator_<sequence_id>" — the viewer join key
    sequence_id: int
    source: str = "pyro-annotator"
    label: str = "unknown"  # unannotated backlog has no ground truth
    camera_id: int | None = None
    camera_name: str | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    started_at: str | None = None
    frames: list[FrameRef] = []


def slugify(value: str | None) -> str:
    """Filesystem-safe lowercase ASCII slug; 'unknown' when nothing remains."""
    ascii_value = (
        unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or "unknown"


def sequence_dir(store_dir: Path, meta: SequenceMeta) -> Path:
    return (
        store_dir
        / slugify(meta.organization_name)
        / slugify(meta.camera_name)
        / f"seq_{meta.sequence_id}"
    )


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_NAME).write_text(meta.model_dump_json(indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    return SequenceMeta.model_validate_json((seq_dir / META_NAME).read_text())


def find_sequence_dirs(store_dir: Path, sequence_id: int) -> list[Path]:
    return sorted(
        p.parent for p in store_dir.glob(f"*/*/seq_{sequence_id}/{META_NAME}")
    )


def sequence_exists(store_dir: Path, sequence_id: int) -> bool:
    return bool(find_sequence_dirs(store_dir, sequence_id))


def iter_sequence_dirs(store_dir: Path) -> Iterator[Path]:
    """Yield every sequence directory under ``store_dir`` (recursive, sorted)."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.glob(f"*/*/seq_*/{META_NAME}")):
        yield meta_path.parent


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_frames(seq_dir: Path, meta: SequenceMeta) -> list[Frame]:
    """Build the ordered ``core`` Frame list the model consumes (meta order = time)."""
    return [
        Frame(
            frame_id=Path(ref.file).stem,
            image_path=seq_dir / ref.file,
            timestamp=_parse_ts(ref.recorded_at),
        )
        for ref in meta.frames
    ]
