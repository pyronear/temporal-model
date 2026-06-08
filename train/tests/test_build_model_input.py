"""Tests for build_model_input CLI helpers + stabilize threading."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.train.build_model_input import _process_one, _to_bool


def test_to_bool_parses_dvc_strings():
    assert _to_bool("true") is True
    assert _to_bool("True") is True
    assert _to_bool("1") is True
    assert _to_bool("false") is False
    assert _to_bool("no") is False


def _write_jpg(path: Path, color: tuple[int, int, int], w: int = 320, h: int = 240):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), color, dtype=np.uint8)).save(path, format="JPEG")


def _write_tube(path: Path, seq_id: str, frame_ids: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sequence_id": seq_id,
                "split": "train",
                "label": "smoke",
                "source": "gt",
                "num_frames": len(frame_ids),
                "tube": {
                    "start_frame": 0,
                    "end_frame": len(frame_ids) - 1,
                    "entries": [
                        {
                            "frame_idx": i,
                            "frame_id": fid,
                            "bbox": [0.3 + 0.2 * i, 0.5, 0.1, 0.1],
                            "is_gap": False,
                            "confidence": 0.9,
                        }
                        for i, fid in enumerate(frame_ids)
                    ],
                },
            }
        )
    )


def test_process_one_threads_stabilize(tmp_path):
    seq_id = "site_1_2023-08-01T00-00-00"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(
            tmp_path / "raw" / "wildfire" / seq_id / "images" / f"{fid}.jpg",
            (200, 30, 30),
        )
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    _write_tube(tube_path, seq_id, frame_ids)
    out_dir = tmp_path / "out"

    sid, label, err = _process_one(
        tube_path,
        tmp_path / "raw",
        out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=True,
    )
    assert err is None and sid == seq_id and label == "smoke"
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is True
