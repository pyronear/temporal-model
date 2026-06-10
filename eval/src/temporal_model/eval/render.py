"""Pure render helpers for the eval viewer (no Streamlit).

Ported from the temporal-model-explorer app's frame/tube drawing layer, repointed
to ``temporal_model.core.crop`` so the displayed tube crops use the same crop
geometry as the classifier — centered on the same stabilized window, but expanded
by a wider display ``CROP_CONTEXT`` (2.0) than the model's 1.5 for legibility.
Kept Streamlit-free so it stays unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from temporal_model.core.crop import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)

CROP_CONTEXT = 2.0  # bbox expansion for tube crops (more context than the model's 1.5)
CROP_SIZE = 224

try:
    _BBOX_FONT = ImageFont.load_default(size=18)  # confidence labels on the frame
except TypeError:  # older Pillow without the size kwarg
    _BBOX_FONT = ImageFont.load_default()

# Display vocabulary. The underlying columns stay label/decision/outcome; the UI
# shows: ground truth (label) · model verdict (decision) · correctness (outcome).
CORRECTNESS = {
    "kept-smoke": "✅ smoke kept",
    "discarded-fp": "✅ fp filtered",
    "discarded-smoke": "🔴 missed smoke",
    "kept-fp": "🟠 false alarm",
    "n/a": "—",
}
ROW_BG = {  # by correctness (errors stand out)
    "🔴 missed smoke": "#f4b4b4",
    "🟠 false alarm": "#fbdca0",
    "✅ smoke kept": "#bfe7bf",
    "✅ fp filtered": "#e6f2e6",
}
KEEP_BG = "#cfe2ff"  # flagged as smoke, ground truth unknown
DISCARD_BG = "#eeeeee"  # discarded, ground truth unknown

TUBE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def day_of(started_at: str | None) -> str:
    """Calendar day (YYYY-MM-DD) from an ISO timestamp; 'unknown' if absent."""
    return started_at[:10] if started_at else "unknown"


def correctness_label(outcome: str) -> str:
    """Human-friendly correctness label for a raw outcome value."""
    return CORRECTNESS.get(outcome, outcome)


def row_background(verdict: str, correctness: str) -> str:
    """Row background colour from the model verdict + correctness label.

    Errors stand out (missed smoke / false alarm); correct rows are green; rows
    with unknown ground truth are tinted by the verdict (kept vs discarded).
    """
    return ROW_BG.get(correctness) or (KEEP_BG if verdict == "keep" else DISCARD_BG)


def tube_color(tube_id: int) -> str:
    """Stable colour for a tube id (same colour in the timeline + crop headers)."""
    return TUBE_PALETTE[tube_id % len(TUBE_PALETTE)]


def legend_html() -> str:
    """HTML chips explaining the table row colours (built from the colour map)."""
    items = [
        ("🔴 missed smoke (real smoke discarded)", ROW_BG["🔴 missed smoke"]),
        ("🟠 false alarm (fp kept)", ROW_BG["🟠 false alarm"]),
        ("✅ smoke kept", ROW_BG["✅ smoke kept"]),
        ("✅ fp filtered", ROW_BG["✅ fp filtered"]),
        ("flagged smoke · GT unknown", KEEP_BG),
        ("discarded · GT unknown", DISCARD_BG),
    ]
    chips = "".join(
        f'<span style="background:{color};color:#111;padding:2px 8px;'
        f'border-radius:4px;margin:0 6px 4px 0;display:inline-block">{label}</span>'
        for label, color in items
    )
    return f'<div style="line-height:2.2">{chips}</div>'


def processed_to_input_index(
    frame_idx: int, padded_frame_indices: list[int]
) -> int | None:
    """Map a model-processed frame index back to the input-frame index.

    The model truncates/pads the sequence; ``padded_frame_indices`` are the
    synthetic (duplicate) slots. A real slot's input index is its position minus
    the number of synthetic slots before it. Returns ``None`` for synthetic slots.
    """
    if frame_idx in padded_frame_indices:
        return None
    return frame_idx - sum(1 for p in padded_frame_indices if p < frame_idx)


def frame_bboxes_by_input_index(details: dict) -> dict[int, list[tuple]]:
    """input-frame index → list of ((cx,cy,w,h), confidence, tube_id) per kept tube."""
    padded = (details or {}).get("preprocessing", {}).get("padded_frame_indices", [])
    out: dict[int, list[tuple]] = {}
    for tube in (details or {}).get("tubes", {}).get("kept", []):
        tid = tube.get("tube_id")
        for entry in tube.get("entries", []):
            if entry.get("bbox") is None:
                continue
            inp = processed_to_input_index(entry["frame_idx"], padded)
            if inp is None:
                continue
            out.setdefault(inp, []).append(
                (tuple(entry["bbox"]), entry.get("confidence"), tid)
            )
    return out


def tube_input_boxes(
    tube: dict, padded_frame_indices: list[int]
) -> list[tuple[int, tuple, float | None]]:
    """(input_index, bbox, confidence) for a tube's real (non-synthetic) entries."""
    boxes: list[tuple[int, tuple, float | None]] = []
    for entry in tube.get("entries", []):
        if entry.get("bbox") is None:
            continue
        inp = processed_to_input_index(entry["frame_idx"], padded_frame_indices)
        if inp is not None:
            boxes.append((inp, tuple(entry["bbox"]), entry.get("confidence")))
    return boxes


def triggering_tube_ids(details: dict) -> set[int]:
    """Tube ids whose full-tube score crosses the decision threshold.

    Each of these tubes would have fired the temporal model on its own (the model
    keeps a sequence when any tube qualifies). For the logistic aggregation the
    score is the calibrated probability; otherwise it's the raw logit — the same
    criterion the model uses to decide.
    """
    dec = (details or {}).get("decision", {})
    threshold = dec.get("threshold")
    if threshold is None:
        return set()
    field = "probability" if dec.get("aggregation") == "logistic" else "logit"
    return {
        tube["tube_id"]
        for tube in (details or {}).get("tubes", {}).get("kept", [])
        if tube.get(field) is not None and tube[field] >= threshold
    }


def _lightning_polygon(x: float, y: float, h: float) -> list[tuple[float, float]]:
    """Points for a small downward lightning bolt with top-left at (x, y).

    Drawn as a polygon because PIL's bundled font has no ⚡ emoji glyph.
    """
    w = h * 0.55
    return [
        (x + 0.55 * w, y),
        (x, y + 0.58 * h),
        (x + 0.40 * w, y + 0.58 * h),
        (x + 0.28 * w, y + h),
        (x + w, y + 0.38 * h),
        (x + 0.55 * w, y + 0.38 * h),
    ]


def draw_bboxes(image_path, boxes, width: int = 4) -> Image.Image:
    """Draw bboxes on the frame; ``boxes`` is ``[(bbox, conf, color, trigger)]``.

    ``trigger`` is ``"decisive"`` (the tube that fired the keep first), ``"would"``
    (a tube that also crosses the threshold), or ``None``. The decisive box gets
    the thickest outline and a filled lightning bolt; a would-trigger box gets a
    medium outline and a hollow bolt. The confidence (when present) is printed
    above the box.
    """
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for (cx, cy, bw, bh), conf, color, trigger in boxes:
        x0, y0 = (cx - bw / 2) * w_img, (cy - bh / 2) * h_img
        x1, y1 = (cx + bw / 2) * w_img, (cy + bh / 2) * h_img
        extra = {"decisive": 3, "would": 1}.get(trigger, 0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width + extra)
        ty = max(0, y0 - 20)
        tx = x0
        if conf is not None:
            label = f"{conf:.2f}"
            draw.text((tx, ty), label, fill=color, font=_BBOX_FONT)
            tx += draw.textlength(label, font=_BBOX_FONT) + 5
        if trigger == "decisive":
            draw.polygon(_lightning_polygon(tx, ty, 16), fill=color)
        elif trigger == "would":
            draw.polygon(_lightning_polygon(tx, ty, 16), outline=color)
    return img


def crop_around_bbox(
    image_path,
    bbox_norm,
    context_factor: float = CROP_CONTEXT,
    patch_size: int = CROP_SIZE,
) -> Image.Image:
    """Square crop centred on a normalized bbox, expanded by ``context_factor``
    (reuses the lib's model-input crop geometry). The default ``CROP_CONTEXT`` (2.0)
    is a bit wider than the model's 1.5, so the crop shows the same window the
    classifier used with a little extra context."""
    img = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = img.shape[:2]
    cx, cy, bw, bh = bbox_norm
    ecx, ecy, ew, eh = expand_bbox(cx, cy, bw, bh, context_factor)
    box = norm_bbox_to_pixel_square(ecx, ecy, ew, eh, img_w, img_h)
    return Image.fromarray(crop_and_resize(img, box, patch_size))


def tube_timeline_df(tube_rows: list[tuple[str, dict]]) -> pd.DataFrame:
    """Long frame for the Altair tube timeline: one row per (tube, present frame).

    ``tube_rows`` is ``[(label, {frame_index: confidence}), ...]``; ``frame_end`` =
    frame + 1 so each present frame renders as a unit-width bar, and ``confidence``
    carries the detector score for the tooltip.
    """
    records = [
        {"tube": label, "frame": f, "frame_end": f + 1, "confidence": conf}
        for label, frames in tube_rows
        for f, conf in sorted(frames.items())
    ]
    return pd.DataFrame(records, columns=["tube", "frame", "frame_end", "confidence"])
