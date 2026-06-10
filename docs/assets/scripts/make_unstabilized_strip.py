"""Render the unstabilized (per-frame box) patch strip for comparison."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire"
MI = ROOT / "train/data/05_model_input/train"
OUT = ROOT / "docs/assets"


def load_font(size):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def unstabilized_strip(seq, idxs, out_name, context_factor=1.5, patch_size=224):
    meta = json.loads((MI / seq / "meta.json").read_text())
    frames = {f["frame_idx"]: f for f in meta["frames"]}
    font = load_font(16)
    tiles = []
    for i in idxs:
        f = frames[i]
        image = np.array(
            Image.open(RAW / seq / "images" / (f["frame_id"] + ".jpg")).convert("RGB")
        )
        img_h, img_w, _ = image.shape
        cx, cy, w, h = expand_bbox(*f["orig_bbox"], context_factor)
        box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch = Image.fromarray(crop_and_resize(image, box, patch_size))
        d = ImageDraw.Draw(patch)
        d.rectangle([0, 0, 64, 24], fill=(0, 0, 0))
        d.text((6, 3), f"t = {i}", fill="white", font=font)
        tiles.append(patch)
    gap = 4
    strip = Image.new(
        "RGB", (patch_size * len(tiles) + gap * (len(tiles) - 1), patch_size), "white"
    )
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += patch_size + gap
    strip.save(OUT / out_name, quality=92)
    print("wrote", OUT / out_name)


unstabilized_strip(
    "pyronear-sdis-07_brison_290_2024-02-03T11-13-08",
    [0, 3, 6, 9, 12, 15, 19],
    "brison_patches_unstabilized.jpg",
)
