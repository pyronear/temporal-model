"""Scan for merge+interp action, bright crops, AND a compact stabilized window."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.tubes import build_tubes, merge_colocated_tubes
from temporal_model.core.inference import filter_and_interpolate_tubes
from temporal_model.core.crop import expand_bbox, norm_bbox_to_pixel_square, crop_and_resize
from temporal_model.core.stabilize import tube_window

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire"
MI = ROOT / "train/data/05_model_input/train"

model = BboxTubeTemporalModel.from_package(ROOT / "api/models/model.zip")
tc = model._cfg["tubes"]
cf = model._cfg["model_input"]["context_factor"]
max_frames = model._cfg["classifier"]["max_frames"]

cands = []
for m in sorted(MI.glob("*/meta.json")):
    d = json.load(open(m))
    if d.get("label") != "smoke":
        continue
    sid = d["sequence_id"]
    try:
        hour = int(sid.rsplit("T", 1)[1][:2])
    except Exception:
        continue
    if not (9 <= hour <= 16):
        continue
    areas = [f["orig_bbox"][2] * f["orig_bbox"][3] for f in d["frames"] if f.get("orig_bbox")]
    if areas and 0.001 < max(areas) < 0.08:
        cands.append(sid)

print(f"{len(cands)} candidates")
hits = 0
for n, sid in enumerate(cands):
    if hits >= 12 or n >= 400:
        break
    img_dir = RAW / sid / "images"
    if not img_dir.exists():
        continue
    paths = sorted(img_dir.glob("*.jpg"))[:max_frames]
    if len(paths) < 12:
        continue
    fd = model.detect(model.load_sequence(paths))
    raw = build_tubes(fd, iou_threshold=tc["iou_threshold"], max_misses=tc["max_misses"])
    f1 = filter_and_interpolate_tubes(
        raw, min_tube_length=tc["infer_min_tube_length"],
        min_detected_entries=tc["min_detected_entries"], interpolate_gaps=False)
    kept = filter_and_interpolate_tubes(
        merge_colocated_tubes(f1, merge_iomin=tc["merge_iomin"],
                              merge_prox_factor=tc["merge_prox_factor"],
                              merge_max_gap=tc["merge_max_gap"]),
        min_tube_length=tc["infer_min_tube_length"],
        min_detected_entries=tc["min_detected_entries"], interpolate_gaps=True)
    if not kept or len(f1) <= len(kept):
        continue
    gappy = max(kept, key=lambda t: sum(e.is_gap for e in t.entries))
    n_gaps = sum(e.is_gap for e in gappy.entries)
    if n_gaps < 2:
        continue
    boxes = [((e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
              if e.detection is not None else None, e.is_gap) for e in gappy.entries]
    win = tube_window(boxes)
    _, _, ww, wh = expand_bbox(*win, cf)
    side = max(ww, wh)  # fraction of frame (approx; aspect ignored)
    if side > 0.35:
        continue
    mid = [e for e in gappy.entries if not e.is_gap and e.detection][len(gappy.entries) // 4]
    img = np.array(Image.open(paths[mid.frame_idx]).convert("RGB"))
    h, w, _ = img.shape
    cx, cy, bw, bh = expand_bbox(mid.detection.cx, mid.detection.cy,
                                 mid.detection.w, mid.detection.h, cf)
    crop = crop_and_resize(img, norm_bbox_to_pixel_square(cx, cy, bw, bh, w, h), 64)
    bright = float(crop.mean())
    if bright < 80:
        continue
    hits += 1
    print(f"bright={bright:5.1f} side={side:.2f} raw={len(raw):2d} f1={len(f1):2d} "
          f"kept={len(kept)} interp={n_gaps:2d}  {sid}")
print("scanned", n + 1)
