"""Tests for TubePatchDataset."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from temporal_model.train.dataset import TubePatchDataset


def _make_split(tmp_path: Path, samples: list[tuple[str, int, int]]) -> Path:
    """Create a fake split dir. samples = [(seq_id, label_int, num_frames), ...]"""
    split = tmp_path / "split"
    split.mkdir()
    index = []
    for seq_id, label_int, num_frames in samples:
        seq_dir = split / seq_id
        seq_dir.mkdir()
        for i in range(num_frames):
            arr = np.full((224, 224, 3), 50 + i, dtype=np.uint8)
            Image.fromarray(arr).save(seq_dir / f"frame_{i:02d}.png")
        meta = {
            "sequence_id": seq_id,
            "split": "train",
            "label": "smoke" if label_int == 1 else "fp",
            "label_int": label_int,
            "num_frames": num_frames,
            "context_factor": 1.5,
            "patch_size": 224,
            "frames": [
                {
                    "frame_idx": i,
                    "frame_id": f"f{i}",
                    "is_gap": False,
                    "orig_bbox": [0.5, 0.5, 0.05, 0.05],
                    "crop_bbox_pixels": [0, 0, 100, 100],
                    "filename": f"frame_{i:02d}.png",
                }
                for i in range(num_frames)
            ],
        }
        (seq_dir / "meta.json").write_text(json.dumps(meta))
        index.append(
            {"sequence_id": seq_id, "label_int": label_int, "num_frames": num_frames}
        )
    (split / "_index.json").write_text(json.dumps(index))
    return split


def test_dataset_length_matches_index(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 5), ("b", 0, 7)])
    ds = TubePatchDataset(split, max_frames=20)
    assert len(ds) == 2


def test_dataset_pads_short_sequences_left_aligned(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 5)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["patches"].shape == (20, 3, 224, 224)
    assert sample["mask"].shape == (20,)
    assert sample["mask"].dtype == torch.bool
    assert sample["mask"][:5].all()
    assert not sample["mask"][5:].any()
    # padded frames are zeros
    assert sample["patches"][5:].abs().sum() == 0


def test_dataset_truncates_too_long_sequences(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 25)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["patches"].shape == (20, 3, 224, 224)
    assert sample["mask"].all()


def test_dataset_returns_label_as_float_tensor(tmp_path):
    split = _make_split(tmp_path, [("smoke_seq", 1, 3), ("fp_seq", 0, 3)])
    ds = TubePatchDataset(split, max_frames=20)
    smoke = ds[0]
    fp = ds[1]
    assert smoke["label"].dtype == torch.float32
    assert smoke["label"].item() == 1.0
    assert fp["label"].item() == 0.0


def test_dataset_normalizes_with_imagenet_stats(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 1)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    # All pixels were value 50 (uint8). After /255 then normalize per channel,
    # the result must NOT equal raw uint8 / 255 (mean/std applied).
    raw = 50.0 / 255.0
    assert not torch.allclose(sample["patches"][0], torch.full((3, 224, 224), raw))


def test_dataset_returns_sequence_id(tmp_path):
    split = _make_split(tmp_path, [("seq42", 1, 3)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["sequence_id"] == "seq42"


def test_dataset_transform_none_preserves_legacy_normalization(tmp_path):
    """transform=None must keep the existing ImageNet-normalize behavior."""
    split = _make_split(tmp_path, [("a", 1, 1)])
    ds = TubePatchDataset(split, max_frames=20, transform=None)
    sample = ds[0]
    raw = 50.0 / 255.0
    assert not torch.allclose(sample["patches"][0], torch.full((3, 224, 224), raw))


def test_dataset_transform_applied_when_provided(tmp_path):
    """When a transform is provided, dataset returns un-normalized [0,1] patches
    into the transform; the transform's output is what the caller sees."""
    split = _make_split(tmp_path, [("a", 1, 3)])

    captured: dict = {}

    def capture(item):
        captured["patches_dtype"] = item["patches"].dtype
        captured["patches_max"] = float(item["patches"].max())
        captured["patches_min"] = float(item["patches"].min())
        captured["mask_sum"] = int(item["mask"].sum())
        item["patches"] = item["patches"] + 1.0  # mutate to prove it flows through
        return item

    ds = TubePatchDataset(split, max_frames=20, transform=capture)
    sample = ds[0]

    # The transform saw un-normalized [0,1] tensors
    assert captured["patches_dtype"] == torch.float32
    assert captured["patches_min"] >= 0.0
    assert captured["patches_max"] <= 1.0
    assert captured["mask_sum"] == 3
    # And its mutation flowed through to the caller
    assert float(sample["patches"].max()) > 1.0
