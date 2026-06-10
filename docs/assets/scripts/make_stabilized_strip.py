"""Stabilized (union-window) patch strip for the crook1 merged tube."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.tubes import build_tubes, merge_colocated_tubes
from temporal_model.core.inference import filter_and_interpolate_tubes
from temporal_model.core.crop import expand_bbox, norm_bbox_to_pixel_square, crop_and_resize
from temporal_model.core.stabilize import tube_window

ROOT = Path(__file__).resolve().parents[3]
SEQ = "pyronear-sdis-07_brison_020_2024-01-18T14-18-41"
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire" / SEQ / "images"
OUT = ROOT / "docs/assets"

model = BboxTubeTemporalModel.from_package(ROOT / "api/models/model.zip")
tc = model._cfg["tubes"]
mi = model._cfg["model_input"]
paths = sorted(RAW.glob("*.jpg"))[: model._cfg["classifier"]["max_frames"]]
fd = model.detect(model.load_sequence(paths))

f1 = filter_and_interpolate_tubes(
    build_tubes(fd, iou_threshold=tc["iou_threshold"], max_misses=tc["max_misses"]),
    min_tube_length=tc["infer_min_tube_length"],
    min_detected_entries=tc["min_detected_entries"], interpolate_gaps=False)
kept = filter_and_interpolate_tubes(
    merge_colocated_tubes(f1, merge_iomin=tc["merge_iomin"],
                          merge_prox_factor=tc["merge_prox_factor"],
                          merge_max_gap=tc["merge_max_gap"]),
    min_tube_length=tc["infer_min_tube_length"],
    min_detected_entries=tc["min_detected_entries"], interpolate_gaps=True)
tube = max(kept, key=lambda t: sum(e.is_gap for e in t.entries))

boxes = [((e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
          if e.detection is not None else None, e.is_gap) for e in tube.entries]
win = tube_window(boxes)
cx, cy, w, h = expand_bbox(*win, mi["context_factor"])

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
SIZE, BORDER = 170, 5
positions = [1, 2, 4, 6, 9, 11, 13]
tiles = []
for pos in positions:
    e = tube.entries[pos]
    t = e.frame_idx
    img = np.array(Image.open(paths[t]).convert("RGB"))
    ih, iw, _ = img.shape
    patch = Image.fromarray(crop_and_resize(img, norm_bbox_to_pixel_square(cx, cy, w, h, iw, ih), SIZE))
    color = (255, 140, 0) if e.is_gap else (60, 179, 113)
    bordered = Image.new("RGB", (SIZE + 2 * BORDER, SIZE + 2 * BORDER), color)
    bordered.paste(patch, (BORDER, BORDER))
    d = ImageDraw.Draw(bordered)
    label = f"t = {t}" + ("  interp." if e.is_gap else "")
    d.rectangle([BORDER, BORDER, BORDER + (118 if e.is_gap else 64), BORDER + 24], fill=(0, 0, 0))
    d.text((BORDER + 6, BORDER + 3), label, fill="white", font=font)
    tiles.append(bordered)
gap = 4
W = tiles[0].width
strip = Image.new("RGB", (W * len(tiles) + gap * (len(tiles) - 1), tiles[0].height), "white")
x = 0
for tl in tiles:
    strip.paste(tl, (x, 0)); x += W + gap
strip.save(OUT / "brison020_patches_stabilized.jpg", quality=92)
print("wrote", OUT / "brison020_patches_stabilized.jpg")
