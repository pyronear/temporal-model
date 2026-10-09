"""Per-tube-entry embeddings from the companion YOLO detector's neck.

The detector's Detect head consumes three feature maps (P3/P4/P5, strides
8/16/32). For each tube entry this module ROI-pools the requested levels at the
entry's box and, optionally, global-pools P5 over the whole frame as scene
context. Training (``train.build_yolo_features``) and inference
(``BboxTubeTemporalModel``) both go through :class:`YoloFeatureExtractor`, so
the two cannot drift.

ponytail: one extra YOLO forward per frame on top of detection (simple and
detector-agnostic); a production path would tap the maps during detection.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import roi_align

__all__ = ["LEVELS", "YoloFeatureExtractor", "letterbox", "model_inputs"]

LEVELS = ("p3", "p4", "p5")  # inputs of the Detect head, in order
ROI_OUTPUT_SIZE = 3  # roi_align grid, mean-pooled to one vector per level
LETTERBOX_FILL = 114  # ultralytics' padding gray


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, dict]:
    """Aspect-preserving resize + centered gray pad to ``size`` x ``size``.

    Same geometry as ultralytics' ``LetterBox(auto=False)``; only consistency
    between training and inference matters here, not bit-parity with it.
    Returns the RGB uint8 array and ``{orig_shape, ratio, left, top}``.
    """
    w0, h0 = image.size
    r = min(size / h0, size / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    left, top = round((size - nw) / 2 - 0.1), round((size - nh) / 2 - 0.1)
    canvas = Image.new("RGB", (size, size), (LETTERBOX_FILL,) * 3)
    canvas.paste(image.resize((nw, nh), Image.BILINEAR), (left, top))
    geom = {"orig_shape": (h0, w0), "ratio": (r, r), "left": left, "top": top}
    return np.asarray(canvas), geom


class YoloFeatureExtractor:
    """ROI + context embeddings from an ultralytics ``YOLO`` detector.

    Args:
        yolo_model: ultralytics ``YOLO`` instance (its ``.model`` is used).
        image_size: Letterbox size, the detector's inference resolution.
        device: Torch device for the forward pass.
    """

    def __init__(self, yolo_model: Any, *, image_size: int, device: Any) -> None:
        self._det = yolo_model.model
        self._device = torch.device(device)
        self._image_size = image_size
        self._strides = [int(s) for s in self._det.stride]
        self._taps: list[torch.Tensor] = []
        self._det.model[-1].register_forward_pre_hook(
            lambda _m, args: self._taps.__setitem__(slice(None), list(args[0]))
        )

    @torch.no_grad()
    def frame_maps(self, image_path: Path) -> tuple[list[torch.Tensor], dict]:
        """Neck maps ``[C, H, W]`` for P3/P4/P5 plus the letterbox geometry."""
        lb, geom = letterbox(Image.open(image_path).convert("RGB"), self._image_size)
        x = torch.from_numpy(np.ascontiguousarray(lb.transpose(2, 0, 1)))
        x = x.to(self._device, torch.float32)[None] / 255
        det = self._det.to(self._device).float().eval()
        det(x)
        return [t[0] for t in self._taps], geom

    def tube_features(
        self,
        entries: list[tuple[Path, tuple[float, float, float, float]]],
        cache: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Per-entry features for one tube: ``(image_path, (cx, cy, w, h))`` list.

        Boxes are normalized to the original frame. Returns float32 arrays with
        one row per entry: ``p3 [K, C3]``, ``p4 [K, C4]``, ``p5 [K, C5]`` and
        ``ctx [K, C5]`` (the entry's whole-frame P5 mean). ``cache`` (path ->
        maps) lets tubes sharing frames reuse one forward per frame.
        """
        cache = {} if cache is None else cache
        rows: dict[str, list[np.ndarray]] = {k: [] for k in (*LEVELS, "ctx")}
        for path, box in entries:
            if path not in cache:
                cache[path] = self.frame_maps(path)
            maps, geom = cache[path]
            h0, w0 = geom["orig_shape"]
            r = geom["ratio"][0]
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * w0 * r + geom["left"]
            y1 = (cy - bh / 2) * h0 * r + geom["top"]
            roi = torch.tensor(
                [[0.0, x1, y1, x1 + bw * w0 * r, y1 + bh * h0 * r]],
                device=maps[0].device,
            )
            for name, fmap, stride in zip(LEVELS, maps, self._strides, strict=True):
                pooled = roi_align(
                    fmap[None].float(),
                    roi,
                    output_size=ROI_OUTPUT_SIZE,
                    spatial_scale=1.0 / stride,
                    sampling_ratio=2,
                    aligned=True,
                )
                rows[name].append(pooled.mean(dim=(2, 3))[0].cpu().numpy())
            rows["ctx"].append(maps[-1].float().mean(dim=(1, 2)).cpu().numpy())
        return {k: np.stack(v).astype(np.float32) for k, v in rows.items()}


def model_inputs(
    feats: dict[str, np.ndarray], *, levels: list[str], max_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(yolo_roi [max_frames, sum C_level], yolo_ctx [max_frames, C5])``.

    Concatenates the requested ``levels`` per entry and zero-pads past the
    tube length (those slots are masked out of attention anyway).
    """
    n = min(len(feats["ctx"]), max_frames)
    roi_dim = sum(feats[lv].shape[1] for lv in levels)
    roi = np.zeros((max_frames, roi_dim), dtype=np.float32)
    ctx = np.zeros((max_frames, feats["ctx"].shape[1]), dtype=np.float32)
    if levels:
        roi[:n] = np.concatenate([feats[lv][:n] for lv in levels], axis=1)
    ctx[:n] = feats["ctx"][:n]
    return roi, ctx
