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
    CLASSIFIER_ONNX_FILENAME,
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
    out = tmp_path / "clf.onnx"
    spec = export_classifier(classifier, out, max_frames=5, patch_size=224)
    assert spec["inputs"]["patches"]["shape"] == ["batch", 5, 3, 224, 224]
    assert spec["inputs"]["mask"]["shape"] == ["batch", 5]
    worst = verify_export(classifier, out, max_frames=5, patch_size=224, atol=1e-4)
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
    assert {MANIFEST_FILENAME, CLASSIFIER_ONNX_FILENAME, CONFIG_FILENAME} <= names
    assert manifest["format_version"] == 1
    assert manifest["model_version"] == "9.9.9"
    assert manifest["variant"] == "test"
    assert manifest["source"]["package"] == model_zip.name
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["onnx"]["inputs"]["patches"]["dtype"] == "float32"
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
