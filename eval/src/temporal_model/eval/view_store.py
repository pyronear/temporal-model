"""Normalized per-sequence record the viewer reads (both source kinds).

One ``sequences/<key>.json`` per scored sequence: identity + metadata + ordered
frame paths (relative to the eval package dir, where dvc repro and the Streamlit
app both run). The viewer joins these to ``results.json`` (scalar verdicts) and
``details/<key>.json`` (tubes) on ``key``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SequenceView:
    key: str
    source: str
    label: str  # "smoke" | "fp" | "unknown"
    organization_name: str | None
    camera_name: str | None
    started_at: str | None
    frames: list[str] = field(default_factory=list)  # paths relative to eval dir


def write_sequence_view(sequences_dir: Path, view: SequenceView) -> None:
    """Write ``sequences/<key>.json`` for one sequence."""
    sequences_dir.mkdir(parents=True, exist_ok=True)
    (sequences_dir / f"{view.key}.json").write_text(json.dumps(asdict(view), indent=2))
