"""Tests for the meta.json sequence-store loader."""

import json
from pathlib import Path

from temporal_model.benchmark.dataset import BenchSequence, iter_sequences


def _write_seq(seq_dir: Path, key: str, label: str, files: list[str]) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        (seq_dir / f).write_bytes(b"x")
    meta = {
        "key": key,
        "label": label,
        "frames": [{"file": f, "created_at": None} for f in files],
    }
    (seq_dir / "meta.json").write_text(json.dumps(meta))


def test_loads_sequences_in_frame_order(tmp_path: Path):
    seq = tmp_path / "org" / "cam" / "seq_1"
    _write_seq(seq, key="seq_1", label="smoke", files=["a.jpg", "b.jpg", "c.jpg"])

    out = list(iter_sequences(tmp_path))

    assert len(out) == 1
    s = out[0]
    assert isinstance(s, BenchSequence)
    assert s.key == "seq_1"
    assert s.label == "smoke"
    assert s.frame_count == 3
    assert [f.image_path.name for f in s.frames] == ["a.jpg", "b.jpg", "c.jpg"]
    assert s.frames[0].image_path == seq / "a.jpg"


def test_finds_sequences_recursively(tmp_path: Path):
    _write_seq(tmp_path / "a" / "s1", "s1", "fp", ["f0.jpg"])
    _write_seq(tmp_path / "b" / "c" / "s2", "s2", "smoke", ["f0.jpg", "f1.jpg"])
    keys = sorted(s.key for s in iter_sequences(tmp_path))
    assert keys == ["s1", "s2"]


def test_missing_store_yields_nothing(tmp_path: Path):
    assert list(iter_sequences(tmp_path / "nope")) == []
