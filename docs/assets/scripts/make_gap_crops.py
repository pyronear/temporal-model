"""Step 3 figure: crops along the merged tube across an interpolated dropout.

Green border = observed detection, orange = interpolated gap entry.
Writes docs/assets/brison020_gap_crops.jpg.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.tubes import build_tubes, merge_colocated_tubes
from temporal_model.core.inference import filter_and_interpolate_tubes
from temporal_model.core.crop import expand_bbox, norm_bbox_to_pixel_square, crop_and_resize

ROOT = Path(__file__).resolve().parents[3]
SEQ = "pyronear-sdis-07_brison_020_2024-01-18T14-18-41"
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire" / SEQ / "images"
OUT = ROOT / "docs/assets"

model = BboxTubeTemporalModel.from_package(ROOT / "api/models/model.zip")
tc = model._cfg["tubes"]
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

gappy = max(kept, key=lambda t: sum(e.is_gap for e in t.entries))

# window: 2 observed before the longest gap run, 3 gaps inside, 1 observed after
runs, cur = [], []
for i, e in enumerate(gappy.entries):
    if e.is_gap:
        cur.append(i)
    elif cur:
        runs.append(cur); cur = []
if cur:
    runs.append(cur)
run = max(runs, key=len)
inside = [run[0], run[len(run) // 2], run[-1]] if len(run) >= 3 else run
picks = [i for i in (run[0] - 2, run[0] - 1) if i >= 0] + inside
nxt = run[-1] + 1
if nxt < len(gappy.entries):
    picks.append(nxt)
window = [gappy.entries[i] for i in picks]

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
SIZE, BORDER = 170, 5
tiles = []
for e in window:
    img = np.array(Image.open(paths[e.frame_idx]).convert("RGB"))
    h, w, _ = img.shape
    cx, cy, bw, bh = expand_bbox(e.detection.cx, e.detection.cy,
                                 e.detection.w, e.detection.h, 1.5)
    patch = Image.fromarray(crop_and_resize(img, norm_bbox_to_pixel_square(cx, cy, bw, bh, w, h), SIZE))
    color = (255, 140, 0) if e.is_gap else (60, 179, 113)
    bordered = Image.new("RGB", (SIZE + 2 * BORDER, SIZE + 2 * BORDER), color)
    bordered.paste(patch, (BORDER, BORDER))
    d = ImageDraw.Draw(bordered)
    label = f"t = {e.frame_idx}" + ("  interp." if e.is_gap else "")
    d.rectangle([BORDER, BORDER, BORDER + (118 if e.is_gap else 64), BORDER + 24], fill=(0, 0, 0))
    d.text((BORDER + 6, BORDER + 3), label, fill="white", font=font)
    tiles.append(bordered)
gap = 4
W = tiles[0].width
strip = Image.new("RGB", (W * len(tiles) + gap * (len(tiles) - 1), tiles[0].height), "white")
x = 0
for t_ in tiles:
    strip.paste(t_, (x, 0)); x += W + gap
strip.save(OUT / "brison020_gap_crops.jpg", quality=92)
print("wrote", OUT / "brison020_gap_crops.jpg")
