"""PyTorch Dataset for the basic temporal smoke classifier.

Reads cropped PNG patches produced by ``scripts/build_model_input.py``
and returns per-tube tensors padded to a fixed length with a mask.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TubePatchDataset(Dataset):
    """Dataset of cropped tube patches stored as PNG folders.

    Each item:

    .. code-block:: python

        {
            "patches": Tensor[max_frames, 3, 224, 224],  # float32
            "mask":    Tensor[max_frames] bool,           # True = real frame
            "label":   Tensor[] float32,                  # 0.0 fp, 1.0 smoke
            "sequence_id": str,
        }

    When ``transform`` is ``None`` (legacy behavior), patches are
    ImageNet-normalized in place. When a ``transform`` callable is provided,
    patches flow into it as un-normalized ``[0, 1]`` tensors and the transform
    is responsible for normalization as its final step.

    Args:
        split_dir: Directory containing ``_index.json`` and one
            sub-directory per tube.
        max_frames: Pad/truncate length.
        transform: Optional callable ``item -> item`` applied after loading.
    """

    def __init__(
        self,
        split_dir: Path,
        max_frames: int,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.max_frames = max_frames
        self.transform = transform
        index = json.loads((self.split_dir / "_index.json").read_text())
        self.index: list[dict] = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        record = self.index[idx]
        seq_id: str = record["sequence_id"]
        label_int: int = record["label_int"]
        seq_dir = self.split_dir / seq_id

        meta = json.loads((seq_dir / "meta.json").read_text())
        frame_files = [seq_dir / f["filename"] for f in meta["frames"]]
        n = min(len(frame_files), self.max_frames)

        patches = torch.zeros(self.max_frames, 3, 224, 224, dtype=torch.float32)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        for i in range(n):
            img = Image.open(frame_files[i]).convert("RGB")
            patches[i] = to_tensor(img)  # CHW float32 in [0, 1]
            mask[i] = True

        item: dict = {
            "patches": patches,
            "mask": mask,
            "label": torch.tensor(float(label_int), dtype=torch.float32),
            "sequence_id": seq_id,
        }

        if self.transform is None:
            # Legacy path: inline ImageNet normalization (valid frames only).
            item["patches"][:n] = (item["patches"][:n] - IMAGENET_MEAN) / IMAGENET_STD
            return item

        return self.transform(item)
