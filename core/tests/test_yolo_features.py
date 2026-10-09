"""Tests for YOLO neck embeddings: extractor, classifier inputs, predict wiring."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_model_edge_cases import TEST_CONFIG, _fake_yolo_factory
from ultralytics.nn.tasks import DetectionModel

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.protocol import Frame
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier
from temporal_model.core.yolo_features import (
    YoloFeatureExtractor,
    letterbox,
    model_inputs,
)

IMG = 128  # tiny letterbox keeps the random YOLO forward fast


@pytest.fixture(scope="module")
def tiny_yolo() -> SimpleNamespace:
    # Random-weight yolo11n built from the bundled yaml: no download needed.
    torch.manual_seed(0)
    return SimpleNamespace(model=DetectionModel("yolo11n.yaml", nc=1, verbose=False))


@pytest.fixture()
def frames(tmp_path: Path) -> list[Frame]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(6):
        p = tmp_path / f"f_{i:02d}.jpg"
        Image.fromarray(rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)).save(p)
        out.append(Frame(frame_id=p.stem, image_path=p, timestamp=None))
    return out


def test_letterbox_geometry_matches_ultralytics():
    from ultralytics.data.augment import LetterBox  # noqa: PLC0415

    img = Image.new("RGB", (1280, 720))
    arr, geom = letterbox(img, 1024)
    ref = LetterBox(new_shape=(1024, 1024), auto=False).get_params(
        {"img": np.zeros((720, 1280, 3), np.uint8)}
    )
    assert arr.shape == (1024, 1024, 3)
    assert (geom["left"], geom["top"]) == (ref["left"], ref["top"])
    assert geom["ratio"] == pytest.approx(ref["ratio"])


def test_tube_features_shapes_and_frame_cache(tiny_yolo, frames):
    ext = YoloFeatureExtractor(tiny_yolo, image_size=IMG, device="cpu")
    cache: dict = {}
    entries = [(f.image_path, (0.5, 0.5, 0.2, 0.3)) for f in frames[:4]]
    feats = ext.tube_features(entries, cache)
    c3, c4, c5 = (feats[k].shape[1] for k in ("p3", "p4", "p5"))
    assert {k: v.shape[0] for k, v in feats.items()} == dict.fromkeys(
        ("p3", "p4", "p5", "ctx"), 4
    )
    assert feats["ctx"].shape[1] == c5 and c3 < c4 < c5
    assert len(cache) == 4
    # A different box on the same frame pools a different region.
    other = ext.tube_features([(frames[0].image_path, (0.2, 0.2, 0.1, 0.1))], cache)
    assert len(cache) == 4
    assert not np.allclose(other["p3"][0], feats["p3"][0])
    np.testing.assert_allclose(other["ctx"][0], feats["ctx"][0])


def test_model_inputs_concat_and_pad():
    feats = {
        "p3": np.ones((3, 2), np.float32),
        "p4": np.full((3, 4), 2, np.float32),
        "p5": np.full((3, 8), 3, np.float32),
        "ctx": np.full((3, 8), 4, np.float32),
    }
    roi, ctx = model_inputs(feats, levels=["p3", "p5"], max_frames=5)
    assert roi.shape == (5, 10) and ctx.shape == (5, 8)
    assert (roi[:3, :2] == 1).all() and (roi[:3, 2:] == 3).all()
    assert (roi[3:] == 0).all() and (ctx[3:] == 0).all()


def test_classifier_without_yolo_has_unchanged_state_dict():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224", pretrained=False, max_frames=4
    )
    assert not clf.uses_yolo
    assert not any("yolo" in k for k in clf.state_dict())


def test_classifier_yolo_only_trains():
    clf = TemporalSmokeClassifier(
        backbone=None,
        yolo_levels=["p3"],
        yolo_roi_dim=16,
        yolo_ctx_dim=8,
        transformer_ffn_dim=32,
        max_frames=4,
    )
    mask = torch.tensor([[True, True, False, False]])
    logit = clf(None, mask, torch.randn(1, 4, 16), torch.randn(1, 4, 8))
    logit.sum().backward()
    assert logit.shape == (1,)
    assert clf.yolo_roi_proj[1].weight.grad is not None


def test_predict_feeds_yolo_features_to_classifier(tiny_yolo, frames):
    det = _fake_yolo_factory([[(0.5, 0.5, 0.2, 0.3, 0.9)] for _ in frames])
    det.model = tiny_yolo.model
    feats = YoloFeatureExtractor(tiny_yolo, image_size=IMG, device="cpu")
    one = feats.tube_features([(frames[0].image_path, (0.5, 0.5, 0.2, 0.3))])
    roi_dim = one["p3"].shape[1] + one["p5"].shape[1]
    ctx_dim = one["ctx"].shape[1]
    cfg = copy.deepcopy(TEST_CONFIG)
    cfg["infer"]["image_size"] = IMG
    clf = TemporalSmokeClassifier(
        backbone=None,
        yolo_levels=["p3", "p5"],
        yolo_roi_dim=roi_dim,
        yolo_ctx_dim=ctx_dim,
        transformer_num_layers=1,
        transformer_ffn_dim=32,
        transformer_dropout=0.0,
        max_frames=6,
    )
    seen = {}
    forward = clf.forward

    def spy(patches, mask, yolo_roi=None, yolo_ctx=None):
        seen.setdefault("shapes", (tuple(yolo_roi.shape), tuple(yolo_ctx.shape)))
        return forward(patches, mask, yolo_roi, yolo_ctx)

    clf.forward = spy
    model = BboxTubeTemporalModel(yolo_model=det, classifier=clf, config=cfg)
    out = model.predict(frames, compute_trigger=True)  # prefix loop needs extras too
    assert seen["shapes"] == ((1, 6, roi_dim), (1, 6, ctx_dim))
    assert len(out.details["tubes"]["kept"]) == 1
