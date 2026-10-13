"""Gradio demo for the two Pyronear smoke models.

Tab 1 — the single-frame detector published at ``pyronear/yolov11s``: one image
in, boxes out.

Tab 2 — the temporal classifier published at ``pyronear/temporal-model``: an
ordered sequence in (frames or a video), one smoke/no-smoke decision out. It
links its detector's boxes into tubes across frames and scores each tube with a
ViT. Its ``model.zip`` bundles its own YOLO, so tab 2's boxes come from that
pinned detector, not from tab 1's.
"""

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

DETECTOR_REPO = os.getenv("DETECTOR_REPO", "pyronear/yolov11s")
# Unset = the repo's main branch, which always holds the latest release.
DETECTOR_REVISION = os.getenv("DETECTOR_REVISION") or None
TEMPORAL_REPO = os.getenv("TEMPORAL_REPO", "pyronear/temporal-model")
TEMPORAL_REVISION = os.getenv("TEMPORAL_REVISION") or None

# The detector settings the temporal pipeline runs with (train/params.yaml).
CONF_THRESHOLD = 0.1
IOU_NMS = 0.2
IMAGE_SIZE = 1024

EXAMPLES_DIR = Path(__file__).parent / "examples"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# High-contrast against sky, smoke and landscape — the eval viewer's matplotlib
# palette disappears on a hazy frame. Tube 0 (usually the only one) is magenta.
TUBE_PALETTE = ["#ff00ff", "#00e5ff", "#ffea00", "#00ff5f", "#ff6d00"]
BOX_COLOR = TUBE_PALETTE[0]
ANIM_MAX_WIDTH = 900
ANIM_FRAME_MS = 600
ANIM_QUALITY = 85

try:
    FONT = ImageFont.load_default(size=18)
except TypeError:  # older Pillow without the size kwarg
    FONT = ImageFont.load_default()


@lru_cache(maxsize=1)
def detector():
    """The single-frame YOLO detector (downloaded once, then cached)."""
    from ultralytics import YOLO  # noqa: PLC0415  # keep app startup fast

    return YOLO(hf_hub_download(DETECTOR_REPO, "best.pt", revision=DETECTOR_REVISION))


@lru_cache(maxsize=1)
def temporal_model():
    """The packaged temporal classifier (downloaded once, then cached)."""
    from temporal_model.core.model import BboxTubeTemporalModel  # noqa: PLC0415

    package = hf_hub_download(TEMPORAL_REPO, "model.zip", revision=TEMPORAL_REVISION)
    return BboxTubeTemporalModel.from_package(Path(package))


def video_to_frames(video_path: str, n_frames: int, out_dir: Path) -> list[Path]:
    """Extract ``n_frames`` evenly spaced frames from a video, in time order."""
    import cv2  # noqa: PLC0415  # only the video path needs OpenCV

    capture = cv2.VideoCapture(video_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise gr.Error(f"Could not read any frame from {Path(video_path).name}")
        step = max(1, total // n_frames)
        indices = list(range(0, total, step))[:n_frames]
        paths = []
        for i, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            path = out_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
    finally:
        capture.release()
    if not paths:
        raise gr.Error(f"Could not decode {Path(video_path).name}")
    return paths


def collect_frames(
    images: list[str] | None, video: str | None, n_frames: int, work_dir: Path
) -> list[Path]:
    """Resolve the sequence input into ordered frame paths.

    Uploaded images are ordered by filename — the Pyronear convention is that
    filename order is time order. A video is sampled into ``n_frames``.
    """
    if video:
        return video_to_frames(video, n_frames, work_dir)
    if not images:
        raise gr.Error("Upload a sequence of frames, or a video.")
    paths = sorted((Path(p) for p in images), key=lambda p: p.name)
    bad = [p.name for p in paths if p.suffix.lower() not in IMAGE_SUFFIXES]
    if bad:
        raise gr.Error(f"Not image files: {', '.join(bad)}")
    return paths


def tube_color(tube_id: int) -> str:
    return TUBE_PALETTE[tube_id % len(TUBE_PALETTE)]


def draw_boxes(image_path: Path | str, boxes: list[tuple]) -> Image.Image:
    """Draw ``[(bbox_cxcywh_normalized, confidence, colour)]`` on a frame.

    The label is the confidence alone. There is one class (smoke), and printing
    its id next to the score read as one number ("0 0.83" ≈ 0.083).
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for (cx, cy, w, h), confidence, colour in boxes:
        x0, y0 = (cx - w / 2) * width, (cy - h / 2) * height
        x1, y1 = (cx + w / 2) * width, (cy + h / 2) * height
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
        if confidence is not None:
            draw.text(
                (x0, max(0, y0 - 20)), f"{confidence:.2f}", fill=colour, font=FONT
            )
    return image


def build_animation(frames: list[Image.Image]) -> str | None:
    """Write the annotated frames as a looping animation; return its path.

    WebP, not GIF: a GIF's 256-colour palette is chosen from the photo, and the
    few hundred magenta box pixels get quantized away to grey.
    """
    if not frames:
        return None
    scale = min(1.0, ANIM_MAX_WIDTH / frames[0].width)
    size = (round(frames[0].width * scale), round(frames[0].height * scale))
    resized = [f.resize(size) for f in frames]
    # ponytail: one file per run left in the OS temp dir (Gradio copies it into
    # its own cache); add explicit cleanup if the Space ever runs out of disk.
    fd, path = tempfile.mkstemp(suffix=".webp")
    os.close(fd)
    resized[0].save(
        path,
        save_all=True,
        append_images=resized[1:],
        duration=ANIM_FRAME_MS,
        quality=ANIM_QUALITY,
        loop=0,
    )
    return path


def input_index(frame_idx: int, padded_frame_indices: list[int]) -> int | None:
    """Map a model-processed frame index back to the uploaded-frame index.

    The model pads short sequences with duplicate frames; ``None`` marks such a
    synthetic slot, which has no uploaded frame to draw on.
    """
    if frame_idx in padded_frame_indices:
        return None
    return frame_idx - sum(1 for p in padded_frame_indices if p < frame_idx)


def detect_one_frame(image_path: str | None):
    """Tab 1: run the single-frame detector and return the annotated image."""
    if not image_path:
        raise gr.Error("Upload an image.")
    results = detector().predict(
        image_path, imgsz=IMAGE_SIZE, conf=CONF_THRESHOLD, iou=IOU_NMS, verbose=False
    )[0]
    confidences = results.boxes.conf.tolist()
    rows = [
        [round(c, 3), *(round(v, 1) for v in xyxy)]
        for c, xyxy in zip(confidences, results.boxes.xyxy.tolist(), strict=True)
    ]
    # Drawn here rather than with results.plot() so both tabs share one style.
    annotated = draw_boxes(
        image_path,
        [
            (tuple(box), conf, BOX_COLOR)
            for box, conf in zip(results.boxes.xywhn.tolist(), confidences, strict=True)
        ],
    )
    summary = f"**{len(rows)} detection(s)**" if rows else "**No detection**"
    return annotated, summary, rows


def classify_sequence(
    images: list[str] | None, video: str | None, n_frames: int, compute_trigger: bool
):
    """Tab 2: run the temporal model and return the decision, frames and tubes."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = collect_frames(images, video, int(n_frames), Path(tmp))
        model = temporal_model()
        output = model.predict(
            model.load_sequence(paths), compute_trigger=compute_trigger
        )
        details = output.details
        kept = details["tubes"]["kept"]
        padded = details["preprocessing"]["padded_frame_indices"]

        boxes_per_frame: dict[int, list[tuple]] = {}
        for tube in kept:
            colour = tube_color(tube["tube_id"])
            for entry in tube["entries"]:
                index = (
                    None
                    if entry["bbox"] is None
                    else input_index(entry["frame_idx"], padded)
                )
                if index is None:
                    continue
                boxes_per_frame.setdefault(index, []).append(
                    (entry["bbox"], entry["confidence"], colour)
                )

        trigger = (
            None
            if output.trigger_frame_index is None
            else input_index(output.trigger_frame_index, padded)
        )
        # The model scores only the first `max_frames`; showing the frames it
        # never looked at would misrepresent the result.
        n_scored = len(paths) - details["preprocessing"]["num_truncated"]
        gallery = []
        for i, path in enumerate(paths[:n_scored]):
            caption = f"{i}: {path.name}"
            if trigger == i:
                caption += " ⚡ trigger"
            gallery.append((draw_boxes(path, boxes_per_frame.get(i, [])), caption))
        animation = build_animation([image for image, _ in gallery])

    probabilities = [t["probability"] for t in kept if t["probability"] is not None]
    verdict = "🔥 **Smoke**" if output.is_positive else "✅ **No smoke**"
    frames_line = f"- frames: {n_scored} scored"
    if n_scored < len(paths):
        frames_line += (
            f", {len(paths) - n_scored} dropped (the model scores the first {n_scored})"
        )
    if padded:
        frames_line += f", {len(padded)} padded"
    lines = [
        verdict,
        f"- probability: **{max(probabilities):.3f}**"
        if probabilities
        else "- probability: n/a (uncalibrated)",
        f"- tubes: {len(kept)} kept / {details['tubes']['num_candidates']} candidates",
        frames_line,
    ]
    if compute_trigger:
        lines.append(
            f"- trigger frame: **{trigger}**"
            if trigger is not None
            else "- trigger frame: none"
        )
    rows = [
        [
            t["tube_id"],
            f"{t['start_frame']}–{t['end_frame']}",
            round(t["logit"], 3),
            None if t["probability"] is None else round(t["probability"], 3),
            t["first_crossing_frame"],
        ]
        for t in kept
    ]
    return "\n".join(lines), animation, gallery, rows, details


def example_sequences() -> list[list]:
    """Any ``examples/<name>/*.jpg`` folder becomes a one-click example."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return [
        [sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)]
        for d in sorted(EXAMPLES_DIR.iterdir())
        if d.is_dir()
    ]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Pyronear — smoke detection") as demo:
        gr.Markdown(
            "# 🔥 Pyronear — smoke detection\n"
            "Two models, two tabs: the **single-frame detector** "
            f"([{DETECTOR_REPO}](https://huggingface.co/{DETECTOR_REPO}) "
            f"`{DETECTOR_REVISION or 'latest'}`) and the **temporal classifier** "
            f"([{TEMPORAL_REPO}](https://huggingface.co/{TEMPORAL_REPO}) "
            f"`{TEMPORAL_REVISION or 'latest'}`), which decides on a whole "
            "sequence.\n\n"
            "Running on free CPU: a few seconds for a short sequence, up to a "
            "minute for a long one with the trigger search on."
        )

        with gr.Tab("Single frame — detection"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="Frame")
                    detect_button = gr.Button("Detect", variant="primary")
                with gr.Column():
                    detect_summary = gr.Markdown()
                    image_out = gr.Image(label="Detections")
            detect_table = gr.Dataframe(
                headers=["confidence", "x0", "y0", "x1", "y1"], label="Boxes (pixels)"
            )
            detect_button.click(
                detect_one_frame,
                inputs=image_in,
                outputs=[image_out, detect_summary, detect_table],
            )

        with gr.Tab("Sequence — temporal model"):
            gr.Markdown(
                "Upload the frames of one sequence (ordered by filename, as in "
                "production) **or** a video, which is sampled into frames. "
                "Boxes are coloured per tube; the animation replays only the "
                "frames the model actually scored."
            )
            with gr.Row():
                with gr.Column():
                    frames_in = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Frames",
                        height=140,  # a long upload list must not push the UI down
                    )
                    video_in = gr.Video(label="…or a video")
                    n_frames_in = gr.Slider(
                        3, 30, value=12, step=1, label="Frames sampled from the video"
                    )
                    trigger_in = gr.Checkbox(
                        label="Compute the trigger frame (slower)", value=False
                    )
                    classify_button = gr.Button("Classify sequence", variant="primary")
                with gr.Column():
                    verdict_out = gr.Markdown()
                    animation_out = gr.Image(
                        label="Scored frames, animated", height=360
                    )
            tubes_out = gr.Dataframe(
                headers=["tube", "frames", "logit", "probability", "crossing"],
                label="Tubes",
            )
            gallery_out = gr.Gallery(
                label="Frame by frame", columns=6, height=300, preview=True
            )
            with gr.Accordion("Raw details", open=False):
                details_out = gr.JSON()
            examples = example_sequences()
            if examples:
                gr.Examples(examples=examples, inputs=frames_in)
            classify_button.click(
                classify_sequence,
                inputs=[frames_in, video_in, n_frames_in, trigger_in],
                outputs=[
                    verdict_out,
                    animation_out,
                    gallery_out,
                    tubes_out,
                    details_out,
                ],
            )

    return demo


if __name__ == "__main__":
    build_demo().launch()
