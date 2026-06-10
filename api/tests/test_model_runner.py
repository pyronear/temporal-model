import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from temporal_model.api import model_runner as mr
from temporal_model.api.model_runner import ModelRunner, read_manifest
from temporal_model.core.protocol import Frame
from temporal_model.core.stage_timer import StageTimer
from temporal_model.core.types import FrameDetections


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


class _OrchestrationModel:
    """Fake core model recording how detection is invoked across calls."""

    def __init__(self):
        self.detect_calls: list[list[str]] = []
        self.predict_calls: list[set[str]] = []
        self.roi_calls: list[tuple | None] = []

    def load_sequence(self, paths):
        return [
            Frame(frame_id=Path(p).stem, image_path=Path(p), timestamp=None)
            for p in paths
        ]

    def detect(self, frames):
        self.detect_calls.append([f.frame_id for f in frames])
        return [
            FrameDetections(
                frame_idx=i, frame_id=f.frame_id, timestamp=None, detections=[]
            )
            for i, f in enumerate(frames)
        ]

    def predict(self, frames, *, frame_detections=None, roi=None, timer=None):
        self.predict_calls.append(set(frame_detections or {}))
        self.roi_calls.append(roi)
        return SimpleNamespace(frame_ids=[f.frame_id for f in frames])


def test_predict_resolves_all_detections_for_model():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    out = asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))

    assert out.frame_ids == ["x_00", "x_01"]
    # predict() receives detections for every frame in the sequence.
    assert model.predict_calls[-1] == {"x_00", "x_01"}


def test_predict_threads_roi_to_model():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"], roi=(0.1, 0.2, 0.3, 0.4)))
    assert model.roi_calls[-1] == (0.1, 0.2, 0.3, 0.4)


def test_predict_roi_defaults_to_none():
    model = _OrchestrationModel()
    runner = ModelRunner(model, name="m", version="1", calibrated=True)
    asyncio.run(runner.predict(["c/x_00.jpg"]))
    assert model.roi_calls[-1] is None


def test_predict_caches_and_reuses_detections():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=4096
    )
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg", "c/x_02.jpg"]))

    assert model.detect_calls[0] == ["x_00", "x_01"]
    assert model.detect_calls[1] == ["x_02"]  # only the new frame re-detected


def test_predict_cache_disabled_detects_every_frame():
    model = _OrchestrationModel()
    runner = ModelRunner(
        model, name="m", version="1", calibrated=True, detection_cache_size=0
    )
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg"]))
    asyncio.run(runner.predict(["c/x_00.jpg", "c/x_01.jpg", "c/x_02.jpg"]))

    assert model.detect_calls[0] == ["x_00", "x_01"]
    assert model.detect_calls[1] == ["x_00", "x_01", "x_02"]  # full each call


class _StubFrame:
    def __init__(self, fid):
        self.frame_id = fid


class _StubModel:
    """Minimal model exposing the load_sequence/detect/predict surface."""

    def __init__(self):
        self.predict_timer = None

    def load_sequence(self, paths):
        return [_StubFrame(str(p)) for p in paths]

    def detect(self, misses):
        return [SimpleNamespace(frame_id=f.frame_id) for f in misses]

    def predict(self, frames, *, frame_detections=None, roi=None, timer=None):
        self.predict_timer = timer
        if timer is not None:
            with timer.stage("classifier"):
                pass
        return SimpleNamespace(is_positive=False, trigger_frame_index=None, details={})


def _runner_with(model, cache_size=4096):
    return ModelRunner(
        model,
        name="m",
        version="1",
        calibrated=False,
        detection_cache_size=cache_size,
    )


def test_predict_records_detector_timing_and_cache_counts():
    model = _StubModel()
    runner = _runner_with(model)
    timer = StageTimer()
    profile: dict = {}

    asyncio.run(runner.predict(["a", "b", "c"], timer=timer, profile=profile))

    timings = timer.as_dict()
    assert "detector" in timings  # detect() was timed
    assert model.predict_timer is timer  # same timer threaded into predict()
    assert profile["n_frames"] == 3
    assert profile["cache_misses"] == 3  # cold: all miss
    assert profile["cache_hits"] == 0


def test_second_predict_hits_cache():
    model = _StubModel()
    runner = _runner_with(model)
    asyncio.run(runner.predict(["a", "b"]))  # warms the cache
    profile: dict = {}
    asyncio.run(runner.predict(["a", "b"], profile=profile))
    assert profile["cache_hits"] == 2
    assert profile["cache_misses"] == 0


def test_predict_without_timer_still_works():
    model = _StubModel()
    runner = _runner_with(model)
    out = asyncio.run(runner.predict(["a"]))
    assert out.is_positive is False
