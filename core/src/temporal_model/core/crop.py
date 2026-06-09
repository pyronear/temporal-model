"""Pure bbox geometry for patch cropping (no I/O).

Used by the inference crop path (``inference.crop_tube_patches``) and the
offline training-data crop (``train.crop_patches.process_tube``); centralized
here so train/inference cropping cannot drift apart.
"""

import numpy as np
from PIL import Image

__all__ = [
    "expand_bbox",
    "norm_bbox_to_pixel_square",
    "crop_and_resize",
]


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
