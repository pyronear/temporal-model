"""Tests for offline crop-patch generation (moved from core.model_input)."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.train.crop_patches import process_tube, save_patch


def _solid_image(w: int, h: int, color: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return arr


def test_save_patch_writes_png_at_target_size(tmp_path):
    img = _solid_image(224, 224, (10, 20, 30))
    out_path = tmp_path / "frame_00.png"
    save_patch(img, out_path)
    assert out_path.is_file()
    loaded = np.array(Image.open(out_path))
    assert loaded.shape == (224, 224, 3)
    assert loaded[0, 0, 0] == 10
    assert loaded[0, 0, 1] == 20
    assert loaded[0, 0, 2] == 30


def _write_jpg(path: Path, color: tuple[int, int, int], w: int = 1280, h: int = 720):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_solid_image(w, h, color)).save(path, format="JPEG", quality=95)


def _write_tube_record(path: Path, sequence_id: str, label: str, frame_ids: list[str]):
    record = {
        "sequence_id": sequence_id,
        "split": "train",
        "label": label,
        "source": "gt",
        "num_frames": len(frame_ids),
        "tube": {
            "start_frame": 0,
            "end_frame": len(frame_ids) - 1,
            "entries": [
                {
                    "frame_idx": i,
                    "frame_id": fid,
                    "bbox": [0.5, 0.5, 0.05, 0.05],
                    "is_gap": False,
                    "confidence": 0.9,
                }
                for i, fid in enumerate(frame_ids)
            ],
        },
    }
    path.write_text(json.dumps(record))
    return record


def test_process_tube_writes_patches_and_meta(tmp_path):
    seq_id = "site_999_2023-05-23T17-18-31"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(3)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))

    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record(tube_path, seq_id, "smoke", frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )

    seq_out = out_dir / seq_id
    assert (seq_out / "frame_00.png").is_file()
    assert (seq_out / "frame_01.png").is_file()
    assert (seq_out / "frame_02.png").is_file()
    meta = json.loads((seq_out / "meta.json").read_text())
    assert meta["sequence_id"] == seq_id
    assert meta["label"] == "smoke"
    assert meta["label_int"] == 1
    assert meta["num_frames"] == 3
    assert meta["context_factor"] == 1.5
    assert meta["patch_size"] == 224
    assert len(meta["frames"]) == 3
    assert meta["frames"][0]["filename"] == "frame_00.png"
    assert meta["frames"][0]["is_gap"] is False


def test_process_tube_uses_filename_from_raw_directory(tmp_path):
    seq_id = "site_999_2023-06-01T10-00-00"
    seq_root = tmp_path / "raw" / "fp" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (10, 20, 30))

    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record(tube_path, seq_id, "fp", frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["label"] == "fp"
    assert meta["label_int"] == 0


def _write_tube_record_2boxes(path: Path, sequence_id: str, frame_ids: list[str]):
    """Tube whose two frames have boxes at different x positions, so the union
    window (cx=0.5) differs from either frame's own box (cx=0.3 / cx=0.7)."""
    boxes = [[0.3, 0.5, 0.1, 0.1], [0.7, 0.5, 0.1, 0.1]]
    record = {
        "sequence_id": sequence_id,
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
                    "bbox": boxes[i],
                    "is_gap": False,
                    "confidence": 0.9,
                }
                for i, fid in enumerate(frame_ids)
            ],
        },
    }
    path.write_text(json.dumps(record))
    return record


def test_process_tube_stabilize_uses_constant_window(tmp_path):
    seq_id = "site_999_2023-07-01T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=True,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    # Same fixed crop window pixels for every frame.
    crop_boxes = [f["crop_bbox_pixels"] for f in meta["frames"]]
    assert crop_boxes[0] == crop_boxes[1]
    # orig_bbox still records each frame's own detection box.
    assert meta["frames"][0]["orig_bbox"] == [0.3, 0.5, 0.1, 0.1]
    assert meta["frames"][1]["orig_bbox"] == [0.7, 0.5, 0.1, 0.1]
    assert meta["stabilize"] is True


def test_process_tube_default_is_stabilized(tmp_path):
    seq_id = "site_999_2023-07-02T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(  # no stabilize arg -> defaults to True
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is True
    assert (
        meta["frames"][0]["crop_bbox_pixels"] == meta["frames"][1]["crop_bbox_pixels"]
    )


def test_process_tube_stabilize_false_is_per_frame(tmp_path):
    seq_id = "site_999_2023-07-03T00-00-00"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record_2boxes(tube_path, seq_id, frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=False,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["stabilize"] is False
    # Different per-frame boxes -> different crop windows.
    assert (
        meta["frames"][0]["crop_bbox_pixels"] != meta["frames"][1]["crop_bbox_pixels"]
    )
