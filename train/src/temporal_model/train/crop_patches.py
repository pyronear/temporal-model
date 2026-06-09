"""Offline training-data crop: tube record -> 224x224 PNG patches + meta.

Moved out of core because this is training-data prep, not a runtime building
block. Shares the crop geometry with the inference path via ``core.crop`` and
the stabilize window policy via ``core.stabilize`` so the two cannot drift.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from temporal_model.core.sequences import find_sequence_dir
from temporal_model.core.stabilize import tube_window

__all__ = ["LABEL_TO_INT", "save_patch", "process_tube"]

LABEL_TO_INT = {"fp": 0, "smoke": 1}


def save_patch(patch: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(path, format="PNG", optimize=True)


def process_tube(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
    stabilize: bool = True,
) -> None:
    record = json.loads(tube_path.read_text())
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        raise FileNotFoundError(f"raw sequence dir not found for {sequence_id}")

    images_dir = seq_dir / "images"
    seq_out = out_dir / sequence_id
    seq_out.mkdir(parents=True, exist_ok=True)

    entries = record["tube"]["entries"]
    window = None
    if stabilize:
        window = tube_window([(tuple(e["bbox"]), e["is_gap"]) for e in entries])

    frame_meta: list[dict] = []
    for entry in entries:
        frame_id = entry["frame_id"]
        frame_idx = entry["frame_idx"]
        bbox = window if stabilize else entry["bbox"]
        is_gap = entry["is_gap"]

        img_path = images_dir / f"{frame_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], context_factor)
        crop_box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch = crop_and_resize(image, crop_box, patch_size)

        filename = f"frame_{frame_idx:02d}.png"
        save_patch(patch, seq_out / filename)

        frame_meta.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "is_gap": is_gap,
                "orig_bbox": list(entry["bbox"]),
                "crop_bbox_pixels": list(crop_box),
                "filename": filename,
            }
        )

    meta = {
        "sequence_id": sequence_id,
        "split": record["split"],
        "label": label,
        "label_int": LABEL_TO_INT[label],
        "num_frames": record["num_frames"],
        "context_factor": context_factor,
        "patch_size": patch_size,
        "stabilize": stabilize,
        "frames": frame_meta,
    }
    (seq_out / "meta.json").write_text(json.dumps(meta, indent=2))
