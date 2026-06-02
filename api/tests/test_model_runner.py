import asyncio
import zipfile
from types import SimpleNamespace

import yaml

from temporal_model.api import model_runner as mr
from temporal_model.api.model_runner import ModelRunner, read_manifest


def _make_package(tmp_path, manifest: dict):
    path = tmp_path / "model.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.yaml", yaml.safe_dump(manifest))
    return path


def test_read_manifest_calibrated(tmp_path):
    path = _make_package(
        tmp_path,
        {
            "variant": "bbox-tube-vit-dinov2",
            "model_version": "1.2.0",
            "logistic_calibrator": "logistic_calibrator.json",
        },
    )
    meta = read_manifest(path)
    assert meta == {
        "name": "bbox-tube-vit-dinov2",
        "version": "1.2.0",
        "calibrated": True,
    }


def test_read_manifest_legacy_uncalibrated(tmp_path):
    path = _make_package(tmp_path, {"variant": "old-model"})
    meta = read_manifest(path)
    assert meta == {"name": "old-model", "version": None, "calibrated": False}


def test_load_uses_lazy_core_model(tmp_path, monkeypatch):
    path = _make_package(
        tmp_path,
        {"variant": "m", "model_version": "9", "logistic_calibrator": "c.json"},
    )
    fake_model = SimpleNamespace(name="fake")
    monkeypatch.setattr(mr, "_load_core_model", lambda p, d: fake_model)

    runner = ModelRunner.load(path, device="cpu")

    assert runner.name == "m"
    assert runner.version == "9"
    assert runner.calibrated is True
    assert runner._model is fake_model


def test_predict_delegates_to_model():
    captured = {}

    class FakeModel:
        def predict_sequence(self, paths):
            captured["paths"] = paths
            return "OUT"

    runner = ModelRunner(FakeModel(), name="m", version="1", calibrated=True)
    result = asyncio.run(runner.predict(["a.jpg", "b.jpg"]))

    assert result == "OUT"
    assert captured["paths"] == ["a.jpg", "b.jpg"]
