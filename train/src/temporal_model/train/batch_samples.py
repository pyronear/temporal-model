"""Dump training-batch visualizations (ultralytics-style) as PNG grids.

Provides :class:`SampleTrainBatchesCallback`, a PyTorch Lightning callback
that writes the first N batches of epoch 0 to disk as
``train_batch_{i}.png`` grids. The intent is to let the user eyeball what
the model actually sees post-augmentation without rerunning training.

Also exposes :func:`render_batch_grid` for standalone use (e.g. from a
notebook or from ``scripts/visualize_augment.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lightning.pytorch import Callback
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.utils import make_grid

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize(x: Tensor) -> Tensor:
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def render_batch_grid(
    patches: Tensor,
    mask: Tensor,
    title: str,
    *,
    denormalize: bool = True,
    row_labels: list[str] | None = None,
) -> Figure:
    """Render a batch of tube patches as a ``B``-row, ``T``-column grid.

    Padded frames are drawn as black cells so the tube length is visible
    at a glance. Adds a subtitle with ``(B, T)`` shape and the per-row
    valid-frame counts, mimicking ultralytics' ``train_batch*.jpg`` layout.

    Args:
        patches:     ``[B, T, 3, H, W]`` batched tube patches.
        mask:        ``[B, T]`` boolean mask (True = real frame).
        title:       Figure title.
        denormalize: If True (default), reverse ImageNet normalization
            (mean/std) before display — use for patches fresh from the
            model's input pipeline. Set False when patches are already
            in ``[0, 1]`` display space.
        row_labels:  Optional per-row labels drawn to the left of the
            grid. Must have length ``B`` if provided.

    Returns:
        A matplotlib ``Figure``. Caller is responsible for closing it.
    """
    if patches.ndim != 5:
        raise ValueError(f"patches must be [B, T, 3, H, W], got {patches.shape}")
    b, t = mask.shape
    if row_labels is not None and len(row_labels) != b:
        raise ValueError(
            f"row_labels length {len(row_labels)} does not match batch size {b}"
        )

    display = patches.detach().float().cpu()
    if denormalize:
        display = _denormalize(display)
    # Zero out padded frames so they show as black in the grid.
    visible = mask.detach().cpu().view(b, t, 1, 1, 1).float()
    display = display * visible
    flat = display.reshape(b * t, *display.shape[2:])
    grid = make_grid(flat, nrow=t, padding=2)

    valid_per_row = mask.detach().cpu().sum(dim=1).tolist()
    full_title = f"{title}  (B={b}, T={t}, valid_frames_per_tube={valid_per_row})"

    fig, ax = plt.subplots(figsize=(max(4.0, t * 1.0), max(2.0, b * 1.0)))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_axis_off()
    ax.set_title(full_title, fontsize=9)

    if row_labels is not None:
        row_height = grid.shape[1] / b
        for i, label in enumerate(row_labels):
            ax.text(
                -8,
                (i + 0.5) * row_height,
                label,
                ha="right",
                va="center",
                fontsize=8,
            )

    return fig


class SampleTrainBatchesCallback(Callback):
    """Save the first ``n_batches`` of epoch 0 as ``train_batch_{i}.png``.

    Produces one PNG per batch under ``output_dir`` showing the full
    batch post-augmentation. Fires only at the start of epoch 0 so it
    adds no cost once the first few batches have been captured.

    Args:
        output_dir: Destination directory (created if missing).
        n_batches:  Number of batches to dump. ``0`` disables the callback.
        dpi:        DPI for the output PNGs.
    """

    def __init__(
        self,
        output_dir: Path | str,
        n_batches: int = 3,
        dpi: int = 120,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_batches = n_batches
        self.dpi = dpi
        self._dumped = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if self.n_batches <= 0:
            return
        if self._dumped >= self.n_batches:
            return
        if trainer.current_epoch > 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        patches = batch["patches"]
        mask = batch["mask"]
        idx = self._dumped
        fig = render_batch_grid(patches, mask, title=f"train_batch_{idx}")
        fig.savefig(
            self.output_dir / f"train_batch_{idx}.png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
        self._dumped += 1
