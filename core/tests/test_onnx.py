"""ONNX export + torch-free runtime: parity with the torch backend."""

import copy
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from temporal_model.core.export_onnx import (
    build_onnx_package,
    export_classifier,
    verify_export,
)
from temporal_model.core.labels import load_frame_detections
from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.onnx_model import (
    BACKBONE_ONNX_FILENAME,
    HEAD_ONNX_FILENAME,
    INPUT_MASK,
    INPUT_PATCHES,
    OUTPUT_LOGIT,
    OnnxTemporalModel,
    UncalibratedModelError,  # via onnx_model: the torch-free import path
    load_onnx_package,
)
from temporal_model.core.package import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    build_model_package,
)
from temporal_model.core.protocol import Frame
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier
from temporal_model.core.types import Detection, FrameDetections

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "wildfire" / "seq_synth01"

# Smallest timm ViT that still exercises the real attention/export path.
CFG: dict = {
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
        "backbone": "vit_tiny_patch16_224",
        "max_frames": 5,
        "pretrained": False,
        "global_pool": "token",
        "transformer_num_layers": 1,
        "transformer_num_heads": 2,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.0,
    },
    "decision": {"aggregation": "max_logit", "threshold": 0.0},
}


@pytest.fixture(scope="module")
def classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    c = CFG["classifier"]
    model = TemporalSmokeClassifier(
        backbone=c["backbone"],
        pretrained=False,
        global_pool=c["global_pool"],
        transformer_num_layers=c["transformer_num_layers"],
        transformer_num_heads=c["transformer_num_heads"],
        transformer_ffn_dim=c["transformer_ffn_dim"],
        transformer_dropout=c["transformer_dropout"],
        max_frames=c["max_frames"],
    )
    return model.eval()


@pytest.fixture(scope="module")
def model_zip(tmp_path_factory: pytest.TempPathFactory, classifier) -> Path:
    root = tmp_path_factory.mktemp("pkg")
    yolo = root / "yolo.pt"
    yolo.write_bytes(b"fake-yolo")
    ckpt = root / "classifier.ckpt"
    torch.save(classifier.state_dict(), ckpt)
    return build_model_package(
        yolo_weights_path=yolo,
        classifier_ckpt_path=ckpt,
        config=copy.deepcopy(CFG),
        variant="test",
        output_path=root / "model.zip",
        model_version="9.9.9",
        allow_uncalibrated=True,
    )


@pytest.fixture(scope="module")
def onnx_zip(tmp_path_factory: pytest.TempPathFactory, model_zip: Path) -> Path:
    out = tmp_path_factory.mktemp("onnx") / "model_onnx.zip"
    return build_onnx_package(model_zip, out, allow_uncalibrated=True)


def _frames() -> list[Frame]:
    return [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]


def _gt_detections() -> dict:
    return {fd.frame_id: fd for fd in load_frame_detections(FIXTURE)}


def test_export_matches_torch_on_every_mask_pattern(classifier, tmp_path: Path):
    backbone, head = tmp_path / "backbone.onnx", tmp_path / "head.onnx"
    spec = export_classifier(classifier, backbone, head, max_frames=5, patch_size=224)
    assert spec["backbone"]["inputs"]["patches"]["shape"] == ["batch", 3, 224, 224]
    assert spec["backbone"]["outputs"]["features"]["shape"] == ["batch", 192]
    assert spec["head"]["inputs"]["features"]["shape"] == ["batch", 5, 192]
    assert spec["head"]["inputs"]["mask"]["shape"] == ["batch", 5]
    worst = verify_export(
        classifier, backbone, head, max_frames=5, patch_size=224, atol=1e-4
    )
    assert worst <= 1e-4
    # Export and verify work on deep copies: the caller's model keeps its
    # as-served configuration (fused attention intact).
    assert all(m.fused_attn for m in classifier.modules() if hasattr(m, "fused_attn"))


def test_uncalibrated_model_zip_is_refused_at_export(model_zip: Path, tmp_path: Path):
    with pytest.raises(UncalibratedModelError):
        build_onnx_package(model_zip, tmp_path / "model_onnx.zip")


def test_onnx_package_contents_and_manifest(model_zip: Path, onnx_zip: Path):
    with zipfile.ZipFile(onnx_zip) as zf:
        names = set(zf.namelist())
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
    assert {
        MANIFEST_FILENAME,
        BACKBONE_ONNX_FILENAME,
        HEAD_ONNX_FILENAME,
        CONFIG_FILENAME,
    } <= names
    assert manifest["format_version"] == 2
    assert manifest["model_version"] == "9.9.9"
    assert manifest["variant"] == "test"
    assert manifest["source"]["package"] == model_zip.name
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["onnx"]["backbone"]["inputs"]["patches"]["dtype"] == "float32"
    assert manifest["onnx"]["head"]["inputs"]["mask"]["dtype"] == "bool"
    assert manifest["onnx"]["max_abs_logit_diff"] <= 1e-4
    assert "logistic_calibrator" not in manifest


def test_uncalibrated_package_is_refused_by_default(onnx_zip: Path):
    with pytest.raises(UncalibratedModelError):
        load_onnx_package(onnx_zip)


def test_predict_parity_with_torch_backend(classifier, onnx_zip: Path):
    frames = _frames()
    dets = _gt_detections()
    torch_model = BboxTubeTemporalModel(
        yolo_model=None, classifier=classifier, config=copy.deepcopy(CFG), device="cpu"
    )
    onnx_model = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)

    ref = torch_model.predict(frames, frame_detections=dets)
    got = onnx_model.predict(frames, frame_detections=dets)

    assert got.is_positive == ref.is_positive
    ref_kept = ref.details["tubes"]["kept"]
    got_kept = got.details["tubes"]["kept"]
    assert len(ref_kept) >= 1
    assert [t["logit"] for t in got_kept] == pytest.approx(
        [t["logit"] for t in ref_kept], abs=1e-4
    )
    strip = lambda kept: [{k: v for k, v in t.items() if k != "logit"} for t in kept]  # noqa: E731
    assert strip(got_kept) == strip(ref_kept)
    assert got.details["preprocessing"] == ref.details["preprocessing"]
    assert got.details["decision"] == ref.details["decision"]


def test_trigger_search_runs_on_onnx_backend(onnx_zip: Path):
    onnx_model = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)
    out = onnx_model.predict(
        _frames(), frame_detections=_gt_detections(), compute_trigger=True
    )
    kept = out.details["tubes"]["kept"]
    assert out.is_positive == any(t["logit"] >= 0.0 for t in kept)
    if out.is_positive:
        assert out.trigger_frame_index is not None


def test_onnx_backend_has_no_detector(onnx_zip: Path):
    onnx_model = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)
    with pytest.raises(NotImplementedError, match="frame_detections"):
        onnx_model.predict(_frames())


def test_onnx_runtime_import_never_loads_torch():
    code = (
        "import sys, temporal_model.core.onnx_model, temporal_model.core.pipeline;"
        "assert 'torch' not in sys.modules, "
        "sorted(m for m in sys.modules if 'torch' in m)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_onnx_score_is_numpy_float32(onnx_zip: Path):
    onnx_model = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)
    patches = np.zeros((2, 5, 3, 224, 224), dtype=np.float32)
    mask = np.ones((2, 5), dtype=bool)
    logits = onnx_model._score(patches, mask)
    assert logits.shape == (2,)
    assert logits.dtype == np.float32
    assert logits[0] == logits[1]


class _CountingSession:
    """Wraps an onnxruntime session, counting runs and the batch sizes seen."""

    def __init__(self, session) -> None:
        self._session = session
        self.batches: list[int] = []

    def run(self, outputs, feeds):
        self.batches.append(len(next(iter(feeds.values()))))
        return self._session.run(outputs, feeds)

    def get_outputs(self):
        return self._session.get_outputs()


def _cached_model(
    onnx_zip: Path, size: int
) -> tuple[OnnxTemporalModel, _CountingSession]:
    model = OnnxTemporalModel.from_package(
        onnx_zip, allow_uncalibrated=True, embedding_cache_size=size
    )
    backbone = _CountingSession(model._backbone)
    model._backbone = backbone
    return model, backbone


def test_cached_predict_matches_and_skips_the_backbone(onnx_zip: Path):
    frames, dets = _frames(), _gt_detections()
    plain = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)
    model, backbone = _cached_model(onnx_zip, size=1000)

    ref = plain.predict(frames, frame_detections=dets)
    first = model.predict(frames, frame_detections=dets)
    n_slots = sum(backbone.batches)
    assert n_slots == sum(
        sum(1 for e in t["entries"] if e["bbox"] is not None)
        for t in ref.details["tubes"]["kept"]
    ), "every real slot embedded exactly once, no padded slot"
    assert (model.cache_hits, model.cache_misses) == (0, n_slots)

    # Same window again: nothing new to embed, same logits.
    again = model.predict(frames, frame_detections=dets)
    assert sum(backbone.batches) == n_slots
    assert model.cache_hits == n_slots
    kept = lambda out: [t["logit"] for t in out.details["tubes"]["kept"]]  # noqa: E731
    assert kept(first) == pytest.approx(kept(ref), abs=1e-5)
    assert kept(again) == kept(first)
    assert first.details == ref.details

    # The trigger search needs the patches back: it takes the uncached path.
    model.predict(frames, frame_detections=dets, compute_trigger=True)
    assert sum(backbone.batches) > n_slots


def test_sliding_window_only_embeds_what_changed(onnx_zip: Path):
    """An engine re-submits its last N frames every round: only new slots cost."""
    frames = _frames()
    model, backbone = _cached_model(onnx_zip, size=1000)
    box = Detection(class_id=0, cx=0.45, cy=0.45, w=0.2, h=0.25, confidence=0.9)
    same_box = {
        f.frame_id: FrameDetections(
            frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[box]
        )
        for i, f in enumerate(frames)
    }
    model.predict(frames[:4], frame_detections=same_box)
    assert backbone.batches == [4]
    # Window slides by one: the stabilized crop window is unchanged (same box
    # everywhere), so the three frames still in it come from the cache.
    out = model.predict(frames[1:], frame_detections=same_box)
    assert backbone.batches == [4, 1]
    assert out.details["tubes"]["kept"][0]["entries"][0]["frame_idx"] == 0

    # The fixture's own boxes drift right: dropping the first frame shrinks the
    # union window, every crop changes, and nothing can be reused (by design:
    # a cached embedding is only valid for the exact crop it was cut from).
    dets = _gt_detections()
    model.predict(frames, frame_detections=dets)
    model.predict(frames[1:], frame_detections=dets)
    assert backbone.batches == [4, 1, 5, 4]


def test_embedding_cache_is_bounded(onnx_zip: Path):
    model, backbone = _cached_model(onnx_zip, size=2)
    frames, dets = _frames(), _gt_detections()
    model.predict(frames, frame_detections=dets)
    assert len(model._cache) == 2
    n = sum(backbone.batches)
    model.predict(frames, frame_detections=dets)
    assert sum(backbone.batches) == n + (n - 2), "only the 2 most recent slots survive"


def test_cache_requires_split_package(onnx_zip: Path):
    pkg = load_onnx_package(onnx_zip, allow_uncalibrated=True)
    with pytest.raises(ValueError, match="format 2"):
        OnnxTemporalModel(
            config=pkg.config, classifier=pkg.head, embedding_cache_size=10
        )
    with pytest.raises(ValueError, match="either"):
        OnnxTemporalModel(config=pkg.config, backbone=pkg.backbone)


@pytest.fixture(scope="module")
def onnx_zip_v1(
    tmp_path_factory: pytest.TempPathFactory, classifier, onnx_zip: Path
) -> Path:
    """A format 1 package (one fused classifier.onnx), as published before format 2."""
    root = tmp_path_factory.mktemp("onnx_v1")
    fused = copy.deepcopy(classifier).cpu().eval()
    for module in fused.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    graph = root / "classifier.onnx"
    torch.onnx.export(
        fused,
        (torch.zeros(2, 5, 3, 224, 224), torch.ones(2, 5, dtype=torch.bool)),
        str(graph),
        input_names=[INPUT_PATCHES, INPUT_MASK],
        output_names=[OUTPUT_LOGIT],
        opset_version=18,
        dynamo=True,
        dynamic_shapes={INPUT_PATCHES: {0: "batch"}, INPUT_MASK: {0: "batch"}},
        external_data=False,
    )
    out = root / "model_onnx.zip"
    with zipfile.ZipFile(onnx_zip) as src, zipfile.ZipFile(out, "w") as dst:
        manifest = yaml.safe_load(src.read(MANIFEST_FILENAME))
        manifest = {
            k: v for k, v in manifest.items() if k not in ("backbone_onnx", "head_onnx")
        }
        manifest.update(format_version=1, classifier_onnx="classifier.onnx")
        dst.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
        dst.write(graph, "classifier.onnx")
        dst.writestr(CONFIG_FILENAME, src.read(CONFIG_FILENAME))
    return out


def test_format_1_package_still_loads_and_agrees(onnx_zip: Path, onnx_zip_v1: Path):
    v1 = OnnxTemporalModel.from_package(onnx_zip_v1, allow_uncalibrated=True)
    v2 = OnnxTemporalModel.from_package(onnx_zip, allow_uncalibrated=True)
    assert not v1.has_split_graphs and v2.has_split_graphs
    frames, dets = _frames(), _gt_detections()
    a = v1.predict(frames, frame_detections=dets).details["tubes"]["kept"]
    b = v2.predict(frames, frame_detections=dets).details["tubes"]["kept"]
    assert [t["logit"] for t in a] == pytest.approx([t["logit"] for t in b], abs=1e-4)
    with pytest.raises(ValueError, match="format 2"):
        OnnxTemporalModel.from_package(
            onnx_zip_v1, allow_uncalibrated=True, embedding_cache_size=10
        )
