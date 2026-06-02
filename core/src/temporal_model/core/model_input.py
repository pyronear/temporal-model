"""Crop tube bboxes from raw frames and save 224x224 PNG patches.

Pure functions for bbox math + crop/save; orchestration lives in
``scripts/build_model_input.py``.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import find_sequence_dir

LABEL_TO_INT = {"fp": 0, "smoke": 1}


def expand_bbox(
    cx: float, cy: float, w: float, h: float, factor: float
) -> tuple[float, float, float, float]:
    return cx, cy, w * factor, h * factor


def norm_bbox_to_pixel_square(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    side_px = max(w * img_w, h * img_h)
    half = side_px / 2.0
    cx_px = cx * img_w
    cy_px = cy * img_h
    x0 = int(round(cx_px - half))
    y0 = int(round(cy_px - half))
    x1 = int(round(cx_px + half))
    y1 = int(round(cy_px + half))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, x1, y1


def crop_and_resize(
    image: np.ndarray, box: tuple[int, int, int, int], patch_size: int
) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1, :]
    h, w, _ = crop.shape
    side = max(h, w)
    if h != w:
        square = np.zeros((side, side, 3), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off : y_off + h, x_off : x_off + w, :] = crop
        crop = square
    pil = Image.fromarray(crop)
    pil = pil.resize((patch_size, patch_size), Image.BILINEAR)
    return np.array(pil)


def save_patch(patch: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(path, format="PNG", optimize=True)


def process_tube(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
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

    frame_meta: list[dict] = []
    for entry in record["tube"]["entries"]:
        frame_id = entry["frame_id"]
        frame_idx = entry["frame_idx"]
        bbox = entry["bbox"]
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
                "orig_bbox": list(bbox),
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
        "frames": frame_meta,
    }
    (seq_out / "meta.json").write_text(json.dumps(meta, indent=2))
