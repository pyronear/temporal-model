"""Parity test: offline training path logits == predict() logits on same input.

Methodology:
- Build tubes offline from GT labels via load_frame_detections + build_tubes
  + select_longest_tube + interpolate_gaps.
- Crop patches via the exact same primitives as TubePatchDataset
  (expand_bbox / norm_bbox_to_pixel_square / crop_and_resize / to_tensor /
  ImageNet normalization), batched in the same shape.
- Forward through the classifier -> reference logit.
- Run BboxTubeTemporalModel.predict() with a fake YOLO that returns the
  same GT detections per frame.
- Assert predict()'s winning-tube logit == reference logit to 1e-5.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from temporal_model.core.labels import load_frame_detections
from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from temporal_model.core.protocol import Frame
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier
from temporal_model.core.tubes import (
    build_tubes,
    interpolate_gaps,
    select_longest_tube,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "wildfire" / "seq_synth01"


def _fake_yolo_from_gt(fixture: Path) -> MagicMock:
    """Build a fake YOLO that returns GT detections per frame."""
    fdets = load_frame_detections(fixture)
    by_path = {}
    for fd in fdets:
        img_path = fixture / "images" / f"{fd.frame_id}.jpg"
        by_path[str(img_path)] = fd.detections

    def predict(paths, **_kwargs):
        results = []
        for p in paths:
            dets = by_path[p]
            r = MagicMock()
            if not dets:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n_dets = len(dets)
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=n_dets: n
                r.boxes.xywhn = torch.tensor([[d.cx, d.cy, d.w, d.h] for d in dets])
                r.boxes.conf = torch.tensor([d.confidence for d in dets])
                r.boxes.cls = torch.tensor([d.class_id for d in dets]).float()
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = predict
    return m


CFG_TRANSFORMER: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 224},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 2,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "vit_small_patch16_224",
        "max_frames": 5,
        "pretrained": False,
        "global_pool": "token",
        "transformer_num_layers": 1,
        "transformer_num_heads": 2,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.0,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture(scope="module")
def transformer_classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    model = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=5,
    )
    model.eval()
    return model


def _offline_logit_with_cfg(classifier: TemporalSmokeClassifier, cfg: dict) -> float:
    """Variant of _offline_logit that reads patch_size/normalization from cfg."""
    fdets = load_frame_detections(FIXTURE)
    tubes = build_tubes(fdets, iou_threshold=0.2, max_misses=2)
    tube = select_longest_tube(tubes)
    assert tube is not None
    interpolate_gaps(tube)

    mi = cfg["model_input"]
    t_max = cfg["classifier"]["max_frames"]
    patches = torch.zeros(t_max, 3, mi["patch_size"], mi["patch_size"])
    mask = torch.zeros(t_max, dtype=torch.bool)
    frame_paths = sorted((FIXTURE / "images").glob("*.jpg"))
    mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)
    window = None
    if mi.get("stabilize", True):
        # Independent oracle: compute the union window from scratch (plain min/max
        # over observed GT boxes) WITHOUT calling the production stabilize helper,
        # so this reference can actually catch a bug in crop_tube_patches' window.
        obs = [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in tube.entries
            if e.detection is not None and not e.is_gap
        ] or [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in tube.entries
            if e.detection is not None
        ]
        x0 = min(cx - bw / 2 for cx, _, bw, _ in obs)
        y0 = min(cy - bh / 2 for _, cy, _, bh in obs)
        x1 = max(cx + bw / 2 for cx, _, bw, _ in obs)
        y1 = max(cy + bh / 2 for _, cy, _, bh in obs)
        window = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        box_src = (
            window if mi.get("stabilize", True) else (det.cx, det.cy, det.w, det.h)
        )
        cx, cy, w, h = expand_bbox(
            box_src[0], box_src[1], box_src[2], box_src[3], mi["context_factor"]
        )
        box = norm_bbox_to_pixel_square(cx, cy, w, h, w_img, h_img)
        p = crop_and_resize(img, box, mi["patch_size"])
        pt = to_tensor(Image.fromarray(p))
        patches[slot] = (pt - mean) / std
        mask[slot] = True

    with torch.no_grad():
        logit = classifier(patches.unsqueeze(0), mask.unsqueeze(0))
    return float(logit.item())


@pytest.mark.parametrize("stabilize", [False, True])
def test_parity_logit_matches_transformer(
    transformer_classifier: TemporalSmokeClassifier, stabilize: bool
) -> None:
    cfg = copy.deepcopy(CFG_TRANSFORMER)
    cfg["model_input"]["stabilize"] = stabilize
    offline = _offline_logit_with_cfg(transformer_classifier, cfg)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    # Pin device to CPU so offline (CPU-only) and online paths share numerics;
    # ViT has enough float32 drift between CPU/GPU to break 1e-5 parity.
    model = BboxTubeTemporalModel(
        yolo_model=yolo,
        classifier=transformer_classifier,
        config=cfg,
        device="cpu",
    )
    out = model.predict(frames=frames)

    kept = out.details["tubes"]["kept"]
    assert len(kept) >= 1
    online = max(t["logit"] for t in kept)

    assert online == pytest.approx(offline, abs=1e-5)
