import logging
from types import SimpleNamespace

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from temporal_model.api.app import _configure_logging, app
from temporal_model.api.model_runner import ModelRunner
from temporal_model.api.settings import settings
from temporal_model.core.package import UncalibratedModelError

BUCKET = "frames"
KEYS = ["cam12/adf_2023-05-23T17-18-01.jpg", "cam12/adf_2023-05-23T17-18-31.jpg"]


def _details(kept, trigger):
    return {
        "decision": {
            "aggregation": "max_logit",
            "threshold": 0.5,
            "trigger_tube_id": trigger,
        },
        "preprocessing": {
            "num_frames_input": 30,
            "num_truncated": 0,
            "padded_frame_indices": [],
        },
        "tubes": {"num_candidates": 2, "num_outside_roi": 0, "kept": kept},
    }


def _smoke_output():
    kept = [
        {
            "tube_id": 7,
            "start_frame": 2,
            "end_frame": 12,
            "logit": 3.4,
            "probability": 0.98,
            "first_crossing_frame": 3,
            "entries": [
                {
                    "frame_idx": 2,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "is_gap": False,
                    "confidence": 0.8,
                }
            ],
        }
    ]
    return SimpleNamespace(
        is_positive=True, trigger_frame_index=3, details=_details(kept, 7)
    )


class FakeRunner:
    name = "bbox-tube-vit-dinov2"
    version = "1.2.0"
    calibrated = True
    threshold_overridden = False
    packaged_threshold = None

    def __init__(self, output=None, error=None):
        self._output = output
        self._error = error
        self.roi = None
        self.bbox = None
        self.bbox_confidence = None

    async def predict(
        self,
        paths,
        *,
        roi=None,
        bbox=None,
        bbox_confidence=1.0,
        timer=None,
        profile=None,
    ):
        self.roi = roi
        self.bbox = bbox
        self.bbox_confidence = bbox_confidence
        if self._error:
            raise self._error
        if timer is not None:
            with timer.stage("detector"):
                pass
        if profile is not None:
            profile.update(n_frames=len(paths), cache_hits=0, cache_misses=len(paths))
        return self._output


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for key in KEYS:
            s3.put_object(Bucket=BUCKET, Key=key, Body=b"\xff\xd8\xff\xe0jpeg")
        with TestClient(app) as c:
            c.app.state.s3_client = s3
            c.app.state.runner = FakeRunner(output=_smoke_output())
            yield c


def test_health_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "bbox-tube-vit-dinov2",
        "model_version": "1.2.0",
    }


def test_predict_requires_token_when_set(client, monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_predict_wrong_token_401(client, monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    r = client.post(
        "/predict",
        json={"frames": KEYS},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized"


def test_predict_correct_token_200(client, monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    r = client.post(
        "/predict",
        json={"frames": KEYS},
        headers={"Authorization": "Bearer s3cr3t"},
    )
    assert r.status_code == 200
    assert r.json()["is_smoke"] is True


def test_predict_open_when_token_unset(client):
    # token defaults to None on the shared settings object → auth off.
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200


def test_health_open_even_with_token_set(client, monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cr3t")
    r = client.get("/health")
    assert r.status_code == 200


def test_lifespan_logs_auth_enabled(monkeypatch, caplog):
    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(settings, "token", "s3cr3t")
    monkeypatch.setattr(
        ModelRunner, "load", lambda *a, **k: FakeRunner(output=_smoke_output())
    )
    with (
        caplog.at_level(logging.INFO, logger="temporal_model.api.app"),
        TestClient(app),
    ):
        pass
    assert "auth enabled" in caplog.text


def test_lifespan_warns_auth_disabled(monkeypatch, caplog):
    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(settings, "token", None)
    monkeypatch.setattr(
        ModelRunner, "load", lambda *a, **k: FakeRunner(output=_smoke_output())
    )
    with (
        caplog.at_level(logging.WARNING, logger="temporal_model.api.app"),
        TestClient(app),
    ):
        pass
    assert "auth disabled" in caplog.text


def test_predict_default(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json() == {
        "is_smoke": True,
        "probability": 0.98,
        "model": {"name": "bbox-tube-vit-dinov2", "version": "1.2.0"},
    }


def test_predict_verbose(client):
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    body = r.json()
    assert r.status_code == 200
    assert body["details"]["preprocessing"]["num_tube_candidates"] == 2
    assert body["details"]["tubes"][0]["tube_id"] == 7
    assert body["is_smoke"] is True
    assert body["probability"] == 0.98
    assert body["model"] == {"name": "bbox-tube-vit-dinov2", "version": "1.2.0"}
    assert body["details"]["decision"] == {
        "aggregation": "max_logit",
        "threshold": 0.5,
        "threshold_overridden": False,
        "packaged_threshold": None,
    }


def test_predict_verbose_surfaces_override(client):
    client.app.state.runner = FakeRunner(output=_smoke_output())
    client.app.state.runner.threshold_overridden = True
    client.app.state.runner.packaged_threshold = 0.5
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    decision = r.json()["details"]["decision"]
    assert decision["threshold_overridden"] is True
    assert decision["packaged_threshold"] == 0.5


def test_health_unavailable(client):
    client.app.state.runner = None
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "model_name": None,
        "model_version": None,
    }


def test_predict_empty_frames_400(client):
    r = client.post("/predict", json={"frames": []})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_scheme_key_400(client):
    r = client.post("/predict", json={"frames": ["s3://frames/a.jpg"]})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_missing_key_404(client):
    r = client.post("/predict", json={"frames": ["cam12/missing.jpg"]})
    assert r.status_code == 404
    assert r.json()["code"] == "frame_not_found"


def test_predict_inference_error_500(client):
    client.app.state.runner = FakeRunner(error=RuntimeError("boom"))
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 500
    assert r.json()["code"] == "inference_error"


def test_predict_model_not_loaded_503(client):
    client.app.state.runner = None
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 503
    assert r.json()["code"] == "model_not_loaded"


def test_predict_mapping_error_returns_coded_500(client):
    # A response-mapping failure (malformed details) must surface as a coded
    # error, not a bare 500 — to_response runs inside the error-handling scope.
    bad = SimpleNamespace(is_positive=True, trigger_frame_index=0, details={})
    client.app.state.runner = FakeRunner(output=bad)
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 500
    assert r.json()["code"] == "inference_error"


def test_predict_uses_request_bucket(monkeypatch):
    # A request-supplied bucket overrides the settings default and is the bucket
    # frames are actually fetched from.
    other = "2eb7ac42fbbf-alert-api-2"
    monkeypatch.setattr(settings, "s3_bucket", "settings-default")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=other)
        for key in KEYS:
            s3.put_object(Bucket=other, Key=key, Body=b"\xff\xd8\xff\xe0jpeg")
        with TestClient(app) as c:
            c.app.state.s3_client = s3
            c.app.state.runner = FakeRunner(output=_smoke_output())
            r = c.post("/predict", json={"frames": KEYS, "bucket": other})
    assert r.status_code == 200
    assert r.json()["is_smoke"] is True


def test_predict_no_bucket_400(client, monkeypatch):
    monkeypatch.setattr(settings, "s3_bucket", "")
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_empty_bucket_400(client):
    r = client.post("/predict", json={"frames": KEYS, "bucket": ""})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_predict_no_bucket_400_takes_precedence_over_model(client, monkeypatch):
    # Missing-bucket validation runs before the model-loaded check, so a request
    # with no bucket is a 400 even when the model is unavailable.
    monkeypatch.setattr(settings, "s3_bucket", "")
    client.app.state.runner = None
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request"


def test_startup_succeeds_without_bucket(monkeypatch):
    # The app no longer hard-requires a settings bucket at startup.
    monkeypatch.setattr(settings, "s3_bucket", "")
    monkeypatch.setattr(
        ModelRunner, "load", lambda *a, **k: FakeRunner(output=_smoke_output())
    )
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True


def test_lifespan_passes_calibrator_threshold(monkeypatch):
    # The lifespan must forward settings.calibrator_threshold into ModelRunner.load
    # (the FakeRunner-injection tests bypass lifespan, so this seam is otherwise
    # untested).
    captured = {}

    def fake_load(
        package_path, device, calibrator_threshold=None, detection_cache_size=0
    ):
        captured["calibrator_threshold"] = calibrator_threshold
        captured["detection_cache_size"] = detection_cache_size
        return FakeRunner(output=_smoke_output())

    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(settings, "calibrator_threshold", 0.33)
    monkeypatch.setattr(settings, "detection_cache_size", 256)
    monkeypatch.setattr(ModelRunner, "load", fake_load)
    with TestClient(app):
        pass
    assert captured["calibrator_threshold"] == 0.33
    assert captured["detection_cache_size"] == 256


def test_configure_logging_idempotent_and_preserves_propagation():
    # Borrow uvicorn's handler without severing propagation (so caplog keeps
    # working) and without duplicating handlers on repeated lifespans.
    app_logger = logging.getLogger("temporal_model")
    uvicorn_logger = logging.getLogger("uvicorn")
    handler = logging.StreamHandler()
    try:
        uvicorn_logger.addHandler(handler)
        _configure_logging()
        _configure_logging()  # idempotent — no duplicate attach
        assert app_logger.propagate is True
        assert app_logger.handlers.count(handler) == 1
        assert app_logger.level == logging.INFO
    finally:
        uvicorn_logger.removeHandler(handler)
        app_logger.removeHandler(handler)


def test_predict_profiling_off_by_default(client):
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    assert r.json()["details"].get("profiling") is None


def test_predict_profiling_on_surfaces_block(client, monkeypatch):
    monkeypatch.setattr(settings, "profile", True)
    r = client.post("/predict?verbose=true", json={"frames": KEYS})
    assert r.status_code == 200
    prof = r.json()["details"]["profiling"]
    assert "s3_fetch" in prof["stages_ms"]
    assert "detector" in prof["stages_ms"]
    assert prof["n_frames"] == len(KEYS)
    assert prof["cache_misses"] == len(KEYS)
    assert prof["total_ms"] >= 0.0


def test_lifespan_uncalibrated_model_degrades_to_unavailable(monkeypatch):
    # An uncalibrated package raises UncalibratedModelError at load (core gate);
    # the lifespan handler must degrade to not-ready rather than crashing.
    def fake_load(*args, **kwargs):
        raise UncalibratedModelError("load_model_package: model is not calibrated")

    monkeypatch.setattr(settings, "s3_bucket", BUCKET)
    monkeypatch.setattr(ModelRunner, "load", fake_load)
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "unavailable"
    assert r.json()["model_loaded"] is False


def test_predict_passes_roi_to_runner(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "roi_xyxyn": [0.1, 0.2, 0.3, 0.4]}
    )
    assert r.status_code == 200
    assert client.app.state.runner.roi == (0.1, 0.2, 0.3, 0.4)


def test_predict_without_roi_passes_none(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert client.app.state.runner.roi is None


def test_predict_invalid_roi_is_400(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "roi_xyxyn": [0.3, 0.2, 0.1, 0.4]}
    )
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_request"
    assert "roi_xyxyn" in body["detail"]


def test_predict_passes_bbox_to_runner(client):
    r = client.post(
        "/predict",
        json={
            "frames": KEYS,
            "bbox_xyxyn": [0.1, 0.2, 0.3, 0.4],
            "bbox_confidence": 0.8,
        },
    )
    assert r.status_code == 200
    assert client.app.state.runner.bbox == (0.1, 0.2, 0.3, 0.4)
    assert client.app.state.runner.bbox_confidence == 0.8


def test_predict_bbox_confidence_defaults_to_one(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "bbox_xyxyn": [0.1, 0.2, 0.3, 0.4]}
    )
    assert r.status_code == 200
    assert client.app.state.runner.bbox_confidence == 1.0


def test_predict_without_bbox_passes_none(client):
    r = client.post("/predict", json={"frames": KEYS})
    assert r.status_code == 200
    assert client.app.state.runner.bbox is None


def test_predict_invalid_bbox_is_400(client):
    r = client.post(
        "/predict", json={"frames": KEYS, "bbox_xyxyn": [0.3, 0.2, 0.1, 0.4]}
    )
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_request"
    assert "bbox_xyxyn" in body["detail"]


def test_predict_bbox_with_roi_is_400(client):
    r = client.post(
        "/predict",
        json={
            "frames": KEYS,
            "bbox_xyxyn": [0.1, 0.2, 0.3, 0.4],
            "roi_xyxyn": [0.0, 0.0, 1.0, 1.0],
        },
    )
    assert r.status_code == 400
    assert "mutually exclusive" in r.json()["detail"]


def test_predict_bbox_confidence_without_bbox_is_400(client):
    r = client.post("/predict", json={"frames": KEYS, "bbox_confidence": 0.9})
    assert r.status_code == 400
    assert "bbox_confidence" in r.json()["detail"]
