"""Sequence store: data/01_raw/sequences/<org>/<camera>/seq_<id>/{meta.json, images/}.

Mirrors the vision-rd explorer's store layout, extended with everything replay
needs: the recorded temporal score + version provenance and, per detection,
the original S3 ``bucket_key`` and the verbatim ``bbox`` string (parsed later
by ``reconstruct``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

META_NAME = "meta.json"
IMAGES_DIR = "images"

# alert-api AnnotationType -> (label, label_detail)
_SMOKE_VALUES = {"wildfire_smoke", "other_smoke"}
_FP_VALUES = {"other"}


class FrameMeta(BaseModel):
    file: str  # relative to the sequence dir, e.g. "images/detection_100.jpg"
    detection_id: int
    created_at: str
    bucket_key: str
    bbox: str  # verbatim alert-api bbox string, e.g. "[(0.1,0.2,0.3,0.4,0.9)]"


class SequenceMeta(BaseModel):
    key: str  # "platform_<sequence_id>" — the viewer join key
    sequence_id: int
    source: str = "platform"
    label: str  # "smoke" | "fp" | "unknown"
    label_detail: str | None = None
    camera_id: int | None = None
    camera_name: str | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    started_at: str | None = None
    temporal_model_score: float | None = None
    temporal_model_version: str | None = None
    temporal_api_version: str | None = None
    frames: list[FrameMeta] = []


def slugify(value: str | None) -> str:
    """Filesystem-safe lowercase slug; 'unknown' when there is nothing to slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return slug or "unknown"


def label_from_is_wildfire(is_wildfire: str | None) -> tuple[str, str | None]:
    """Map alert-api's ``is_wildfire`` annotation to (label, label_detail)."""
    if is_wildfire in _SMOKE_VALUES:
        return "smoke", is_wildfire
    if is_wildfire in _FP_VALUES:
        return "fp", is_wildfire
    return "unknown", is_wildfire


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


def sequence_exists(store_dir: Path, sequence_id: int) -> bool:
    return any(store_dir.glob(f"*/*/seq_{sequence_id}/{META_NAME}"))


def iter_metas(store_dir: Path) -> Iterator[tuple[Path, SequenceMeta]]:
    """Yield (sequence_dir, meta) for every sequence in the store."""
    for meta_path in sorted(store_dir.glob(f"*/*/seq_*/{META_NAME}")):
        yield meta_path.parent, read_meta(meta_path.parent)
