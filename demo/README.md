---
title: Pyronear Wildfire Detection
emoji: 🔥
colorFrom: blue
colorTo: pink
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - pyronear/yolov11s
  - pyronear/temporal-model
---

# Pyronear — smoke detection demo

Gradio Space to try the two Pyronear wildfire-smoke models side by side. Source
lives in [`pyronear/temporal-model`](https://github.com/pyronear/temporal-model)
under `demo/`; the Space is a mirror of that folder.

| Tab | Model | Input | Output |
|---|---|---|---|
| Single frame — detection | [`pyronear/yolov11s`](https://huggingface.co/pyronear/yolov11s) latest | one image | boxes + confidences |
| Sequence — temporal model | [`pyronear/temporal-model`](https://huggingface.co/pyronear/temporal-model) latest | ordered frames, or a video | smoke / no-smoke, per-tube probabilities, trigger frame |

Both tabs run the detector at the pipeline's production settings
(`conf=0.1`, `iou=0.2`, `imgsz=1024`). The temporal `model.zip` bundles its own
pinned YOLO, so tab 2's boxes come from that one — they can differ from tab 1's.

Frames of a sequence are ordered **by filename**, the production convention
(`<prefix>_<YYYY-MM-DDTHH-MM-SS>.jpg`). A video is sampled into evenly spaced
frames instead.

The sequence tab answers with an animated WebP of the scored frames plus a
frame-by-frame carousel under it. The model scores only the first `max_frames`
(20) of a sequence, so the extra frames are reported as dropped and are shown in
neither view — displaying frames the model never looked at would misrepresent
the result. (WebP rather than GIF: a GIF's 256-colour palette is picked from the
photo and quantizes the magenta boxes away to grey.)

The trigger frame (earliest frame the model would have fired on) is off by
default: it re-scores growing prefixes of each tube, which is slow on CPU.

## Run locally

```bash
make install
make run          # http://localhost:7860
```

`make install` builds `.venv` from `requirements.txt`, which pulls
`temporal-model-core[torch]` (torch, timm, ultralytics) — the first run is a
large download. Model weights are fetched from the Hub on first use and cached.

Both tabs follow the latest release of each repo (its `main` branch). Pin a
version or switch repo with `DETECTOR_REPO` / `DETECTOR_REVISION` /
`TEMPORAL_REPO` / `TEMPORAL_REVISION`.

## Examples

Drop a folder of frames in `demo/examples/<name>/` and it shows up as a
one-click example in the sequence tab. Committed examples are mirrored to the
Space; none are bundled today.

## Deploy

```bash
hf auth login    # once, or set HF_TOKEN
make push
```

Mirrors this folder to the Space, deleting anything there that is no longer
here.
