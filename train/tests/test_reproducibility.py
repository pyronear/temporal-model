"""End-to-end reproducibility test for the training pipeline.

Runs two short Lightning fits with the same seed on a tiny fake dataset and
asserts that every weight in the final ``state_dict`` is bitwise identical.
A third run with a different seed acts as a negative control so the test
cannot silently pass if nothing is actually random. Runs on CPU always and
on GPU when CUDA is available (skipped otherwise, e.g. in CI).

Exercises the full seeding path used by ``scripts/train.py``:
``L.seed_everything(seed, workers=True)`` + ``Trainer(deterministic=True)``
with ``num_workers > 0`` so Lightning's auto-injected ``worker_init_fn`` is
in play.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from temporal_model.train.dataset import TubePatchDataset
from temporal_model.train.lit_temporal import LitTemporalClassifier

SEED = 1234
OTHER_SEED = 5678


def _make_split(
    root: Path, split_name: str, samples: list[tuple[str, int, int]]
) -> Path:
    split = root / split_name
    split.mkdir()
    index = []
    for seq_id, label_int, num_frames in samples:
        seq_dir = split / seq_id
        seq_dir.mkdir()
        for i in range(num_frames):
            arr = np.full((224, 224, 3), 50 + i * 5, dtype=np.uint8)
            Image.fromarray(arr).save(seq_dir / f"frame_{i:02d}.png")
        meta = {
            "sequence_id": seq_id,
            "split": split_name,
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


def _fit_once_transformer(
    seed: int, train_dir: Path, val_dir: Path, log_dir: Path, accelerator: str
) -> dict:
    L.seed_everything(seed, workers=True)

    train_ds = TubePatchDataset(train_dir, max_frames=5)
    val_ds = TubePatchDataset(val_dir, max_frames=5)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    lit = LitTemporalClassifier(
        backbone="vit_small_patch16_224",
        learning_rate=1e-4,
        weight_decay=5e-2,
        pretrained=False,
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=5,
        global_pool="token",
    )

    trainer = L.Trainer(
        max_epochs=2,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        default_root_dir=log_dir,
    )
    trainer.fit(lit, train_loader, val_loader)
    return {k: v.detach().clone() for k, v in lit.state_dict().items()}


def _assert_bitwise_reproducible(
    tmp_path: Path, accelerator: str, *, negative_control: bool = True
) -> None:
    train_dir = _make_split(
        tmp_path,
        "train",
        [("a", 1, 5), ("b", 0, 4), ("c", 1, 3), ("d", 0, 5)],
    )
    val_dir = _make_split(tmp_path, "val", [("e", 1, 4), ("f", 0, 3)])

    run1 = _fit_once_transformer(
        SEED, train_dir, val_dir, tmp_path / "run1", accelerator
    )
    run2 = _fit_once_transformer(
        SEED, train_dir, val_dir, tmp_path / "run2", accelerator
    )

    assert run1.keys() == run2.keys()
    for key in run1:
        assert torch.equal(run1[key], run2[key]), (
            f"Same-seed transformer runs diverged at {key!r}"
        )

    if negative_control:
        run_other = _fit_once_transformer(
            OTHER_SEED, train_dir, val_dir, tmp_path / "run_other", accelerator
        )
        assert run_other.keys() == run1.keys()
        differing = [
            key for key in run1 if not torch.equal(run1[key], run_other[key])
        ]
        assert differing, "Different-seed run produced identical transformer weights"


def test_transformer_training_is_bitwise_reproducible_with_fixed_seed(
    tmp_path: Path,
) -> None:
    _assert_bitwise_reproducible(tmp_path, accelerator="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_transformer_training_is_bitwise_reproducible_on_gpu(tmp_path: Path) -> None:
    """GPU twin of the CPU test.

    ``Trainer(deterministic=True)`` enables strict
    ``torch.use_deterministic_algorithms`` and sets
    ``CUBLAS_WORKSPACE_CONFIG``, so CUDA kernels must be deterministic too.
    Guards against changes (e.g. mixed precision, attention backends) that
    would silently break GPU run-to-run reproducibility.

    Skips the different-seed negative control: seeds diverge the model at
    init on the CPU before the GPU is involved, so the control proves
    nothing GPU-specific — and the CPU test always runs it.
    """
    _assert_bitwise_reproducible(tmp_path, accelerator="gpu", negative_control=False)
