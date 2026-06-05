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
    assert runner.threshold_overridden is False
    assert runner.packaged_threshold is None


class _FakeModel:
    """Fake core model exposing real logistic_threshold and aggregation."""

    def __init__(self, threshold=0.5, aggregation="logistic"):
        self._t = threshold
        self.aggregation = aggregation

    @property
    def logistic_threshold(self):
        return self._t

    @logistic_threshold.setter
    def logistic_threshold(self, value):
        self._t = value


def test_load_applies_override_when_logistic(tmp_path, monkeypatch):
    path = _make_package(tmp_path, {"variant": "m", "logistic_calibrator": "c.json"})
    model = _FakeModel(threshold=0.5, aggregation="logistic")
    monkeypatch.setattr(mr, "_load_core_model", lambda p, d: model)

    runner = ModelRunner.load(path, device="cpu", calibrator_threshold=0.8)

    assert model.logistic_threshold == 0.8
    assert runner.threshold_overridden is True
    assert runner.packaged_threshold == 0.5


def test_load_ignores_override_when_not_logistic(tmp_path, monkeypatch, caplog):
    # Calibrator absent and/or max_logit decision: the override does not apply.
    path = _make_package(tmp_path, {"variant": "old-model"})
    model = _FakeModel(threshold=0.5, aggregation="max_logit")
    monkeypatch.setattr(mr, "_load_core_model", lambda p, d: model)

    with caplog.at_level("WARNING"):
        runner = ModelRunner.load(path, device="cpu", calibrator_threshold=0.8)

    assert model.logistic_threshold == 0.5
    assert runner.threshold_overridden is False
    assert runner.packaged_threshold is None
    assert "logistic" in caplog.text


def test_load_no_override_leaves_threshold(tmp_path, monkeypatch):
    path = _make_package(tmp_path, {"variant": "m", "logistic_calibrator": "c.json"})
    model = _FakeModel(threshold=0.5, aggregation="logistic")
    monkeypatch.setattr(mr, "_load_core_model", lambda p, d: model)

    runner = ModelRunner.load(path, device="cpu")

    assert model.logistic_threshold == 0.5
    assert runner.threshold_overridden is False


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
