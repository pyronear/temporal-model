"""Tests for training-batch sampling callback + renderer."""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from temporal_model.train.batch_samples import (
    SampleTrainBatchesCallback,
    render_batch_grid,
)


def _fake_batch(b: int, t: int, n_valid: int, h: int = 32, w: int = 32) -> dict:
    patches = torch.rand(b, t, 3, h, w) * 0.5 + 0.25  # roughly in [0.25, 0.75]
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, :n_valid] = True
    return {"patches": patches, "mask": mask}


def test_render_batch_grid_returns_figure_with_expected_axes():
    batch = _fake_batch(b=2, t=4, n_valid=3)
    fig = render_batch_grid(batch["patches"], batch["mask"], title="x")
    try:
        assert isinstance(fig, Figure)
        # Exactly one axes, no xticks/yticks (axis off).
        axes = fig.get_axes()
        assert len(axes) == 1
        # Title now carries a (B, T, valid_frames_per_tube) suffix, so
        # just check the user-supplied prefix is preserved.
        title = axes[0].get_title()
        assert title.startswith("x")
        assert "B=2" in title and "T=4" in title
    finally:
        plt.close(fig)


def test_render_batch_grid_rejects_wrong_rank():
    bad = torch.zeros(3, 3, 32, 32)  # [T, 3, H, W] missing batch dim
    mask = torch.ones(1, 3, dtype=torch.bool)
    try:
        render_batch_grid(bad, mask, title="bad")
    except ValueError as e:
        assert "patches must be" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_render_batch_grid_row_labels_require_matching_length():
    batch = _fake_batch(b=2, t=4, n_valid=3)
    try:
        render_batch_grid(batch["patches"], batch["mask"], title="x", row_labels=["a"])
    except ValueError as e:
        assert "row_labels" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_render_batch_grid_denormalize_toggle_affects_pixels():
    """denormalize=True should produce visibly different output than
    denormalize=False for identical input."""
    batch = _fake_batch(b=2, t=3, n_valid=3)
    fig_denorm = render_batch_grid(
        batch["patches"], batch["mask"], title="a", denormalize=True
    )
    fig_raw = render_batch_grid(
        batch["patches"], batch["mask"], title="b", denormalize=False
    )
    try:
        img_denorm = fig_denorm.axes[0].images[0].get_array()
        img_raw = fig_raw.axes[0].images[0].get_array()
        # The inputs are roughly in [0.25, 0.75]; after ImageNet-denorm
        # they spread further, so the two arrays cannot be identical.
        assert not torch.allclose(torch.as_tensor(img_denorm), torch.as_tensor(img_raw))
    finally:
        plt.close(fig_denorm)
        plt.close(fig_raw)


def test_callback_dumps_n_batches_then_stops(tmp_path: Path):
    out = tmp_path / "batch_samples"
    cb = SampleTrainBatchesCallback(output_dir=out, n_batches=2)
    trainer = SimpleNamespace(current_epoch=0)

    # Fire 5 times; only 2 PNGs should land on disk.
    for i in range(5):
        batch = _fake_batch(b=2, t=4, n_valid=3)
        cb.on_train_batch_start(trainer, pl_module=None, batch=batch, batch_idx=i)

    pngs = sorted(out.glob("*.png"))
    assert [p.name for p in pngs] == ["train_batch_0.png", "train_batch_1.png"]


def test_callback_noop_after_epoch_zero(tmp_path: Path):
    out = tmp_path / "batch_samples"
    cb = SampleTrainBatchesCallback(output_dir=out, n_batches=3)
    trainer = SimpleNamespace(current_epoch=1)

    batch = _fake_batch(b=2, t=4, n_valid=3)
    cb.on_train_batch_start(trainer, pl_module=None, batch=batch, batch_idx=0)

    assert not out.exists() or not any(out.glob("*.png"))


def test_callback_disabled_when_n_batches_zero(tmp_path: Path):
    out = tmp_path / "batch_samples"
    cb = SampleTrainBatchesCallback(output_dir=out, n_batches=0)
    trainer = SimpleNamespace(current_epoch=0)

    batch = _fake_batch(b=2, t=4, n_valid=3)
    cb.on_train_batch_start(trainer, pl_module=None, batch=batch, batch_idx=0)

    assert not out.exists() or not any(out.glob("*.png"))
