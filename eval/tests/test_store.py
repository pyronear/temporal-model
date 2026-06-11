import json
from pathlib import Path

from temporal_model.eval.store import (
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    normalize_label,
    read_meta,
)


def _write_store_seq(root: Path, key: str, label: str) -> Path:
    seq = root / "org-a" / "cam-1" / key
    (seq / "images").mkdir(parents=True)
    (seq / "images" / "f0.jpg").write_bytes(b"\xff")
    (seq / "images" / "f1.jpg").write_bytes(b"\xff")
    meta = {
        "key": key,
        "sequence_id": key,
        "source": "pyro-annotator",
        "label": label,
        "label_detail": None,
        "label_source": "pyro_annotator_folder",
        "frames": [
            {"file": "images/f0.jpg", "detection_id": None, "created_at": None},
            {"file": "images/f1.jpg", "detection_id": None, "created_at": None},
        ],
        "camera_id": 1,
        "camera_name": "cam-1",
        "organization_id": 7,
        "organization_name": "org-a",
        "started_at": "2026-05-19T14:10:01",
    }
    (seq / "meta.json").write_text(json.dumps(meta))
    return seq


def test_iter_and_read_meta(tmp_path):
    _write_store_seq(tmp_path, "seq-1", "smoke")
    dirs = list(iter_sequence_dirs(tmp_path))
    assert len(dirs) == 1
    meta = read_meta(dirs[0])
    assert isinstance(meta, SequenceMeta)
    assert meta.key == "seq-1"
    assert meta.label == "smoke"
    assert meta.organization_name == "org-a"
    assert len(meta.frames) == 2


def test_build_frames_orders_by_meta(tmp_path):
    seq = _write_store_seq(tmp_path, "seq-1", "fp")
    meta = read_meta(seq)
    frames = build_frames(seq, meta)
    assert [f.image_path.name for f in frames] == ["f0.jpg", "f1.jpg"]


def test_normalize_label():
    assert normalize_label("wildfire", ["wildfire"], ["false_positive"]) == "smoke"
    assert normalize_label("false_positive", ["wildfire"], ["false_positive"]) == "fp"
    assert normalize_label(None, [], []) == "unknown"
