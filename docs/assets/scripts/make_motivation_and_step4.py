"""Render README illustration candidates from train data sequences."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire"
MI = ROOT / "train/data/05_model_input/train"
OUT = ROOT / "docs/assets"

RED = (230, 40, 40)
ORANGE = (255, 165, 0)


def load_font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_strip(seq, idxs, out_name, width=480):
    meta = json.loads((MI / seq / "meta.json").read_text())
    frames = {f["frame_idx"]: f for f in meta["frames"]}
    tiles = []
    font = load_font(22)
    for i in idxs:
        f = frames[i]
        img = Image.open(RAW / seq / "images" / (f["frame_id"] + ".jpg")).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        if f.get("orig_bbox"):
            cx, cy, w, h = f["orig_bbox"]
            d.rectangle(
                [
                    (cx - w / 2) * W,
                    (cy - h / 2) * H,
                    (cx + w / 2) * W,
                    (cy + h / 2) * H,
                ],
                outline=RED,
                width=6,
            )
        img = img.resize((width, int(width * H / W)), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        label = f"t = {i}"
        d.rectangle([0, 0, 110, 34], fill=(0, 0, 0))
        d.text((10, 4), label, fill="white", font=font)
        tiles.append(img)
    gap = 6
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    strip = Image.new("RGB", (total_w, tiles[0].height), "white")
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + gap
    strip.save(OUT / out_name, quality=92)


def annotated_frame(seq, idx, out_name):
    meta = json.loads((MI / seq / "meta.json").read_text())
    f = {fr["frame_idx"]: fr for fr in meta["frames"]}[idx]
    img = Image.open(RAW / seq / "images" / (f["frame_id"] + ".jpg")).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    if f.get("crop_bbox_pixels"):
        x0, y0, x1, y1 = f["crop_bbox_pixels"]
        d.rectangle([x0, y0, x1, y1], outline=ORANGE, width=5)
    if f.get("orig_bbox"):
        cx, cy, w, h = f["orig_bbox"]
        d.rectangle(
            [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H],
            outline=RED,
            width=5,
        )
    font = load_font(28)
    d.text(
        (12, H - 80),
        "red: YOLO detection",
        fill=RED,
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    d.text(
        (12, H - 44),
        "orange: stabilized crop window (union x context)",
        fill=ORANGE,
        font=font,
        stroke_width=2,
        stroke_fill="black",
    )
    img.save(OUT / out_name, quality=92)


def patch_strip(seq, idxs, out_name):
    font = load_font(16)
    tiles = []
    for i in idxs:
        p = Image.open(MI / seq / f"frame_{i:02d}.png").convert("RGB")
        d = ImageDraw.Draw(p)
        d.rectangle([0, 0, 64, 24], fill=(0, 0, 0))
        d.text((6, 3), f"t = {i}", fill="white", font=font)
        tiles.append(p)
    gap = 4
    strip = Image.new("RGB", (224 * len(tiles) + gap * (len(tiles) - 1), 224), "white")
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += 224 + gap
    strip.save(OUT / out_name, quality=92)


for seq, tag in [
    ("pyronear-sdis-07_brison_290_2024-02-03T11-13-08", "brison"),
    ("adf_avinyonet_999_2023-05-23T17-18-31", "avinyonet"),
]:
    frame_strip(seq, [0, 6, 12, 19], f"{tag}_frame_strip.jpg")
    annotated_frame(seq, 19, f"{tag}_annotated.jpg")
    patch_strip(seq, [0, 3, 6, 9, 12, 15, 19], f"{tag}_patches.jpg")
    print("done", tag)
