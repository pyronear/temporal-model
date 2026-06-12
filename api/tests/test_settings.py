import pytest
from pydantic import ValidationError

from temporal_model.api.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model_path == "/models/model.zip"
    assert s.device is None
    assert s.calibrator_threshold is None
    assert s.s3_bucket == ""
    assert s.s3_region is None
    assert s.s3_endpoint_url is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_S3_BUCKET", "pyro-frames")
    monkeypatch.setenv("TEMPORAL_API_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("TEMPORAL_API_DEVICE", "cpu")
    s = Settings(_env_file=None)
    assert s.s3_bucket == "pyro-frames"
    assert s.s3_endpoint_url == "http://minio:9000"
    assert s.device == "cpu"


def test_calibrator_threshold_parsed(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_CALIBRATOR_THRESHOLD", "0.8")
    assert Settings(_env_file=None).calibrator_threshold == 0.8


@pytest.mark.parametrize("value", ["0.0", "1.0"])
def test_calibrator_threshold_accepts_boundaries(monkeypatch, value):
    monkeypatch.setenv("TEMPORAL_API_CALIBRATOR_THRESHOLD", value)
    assert Settings(_env_file=None).calibrator_threshold == float(value)


@pytest.mark.parametrize("value", ["1.5", "-0.1"])
def test_calibrator_threshold_out_of_range_fails(monkeypatch, value):
    monkeypatch.setenv("TEMPORAL_API_CALIBRATOR_THRESHOLD", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_detection_cache_size_default():
    assert Settings(_env_file=None).detection_cache_size == 4096


def test_detection_cache_size_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_DETECTION_CACHE_SIZE", "10")
    assert Settings(_env_file=None).detection_cache_size == 10


def test_api_token_default_none():
    assert Settings(_env_file=None).token is None


def test_api_token_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_TOKEN", "s3cr3t")
    assert Settings(_env_file=None).token == "s3cr3t"


def test_frame_source_default_s3():
    assert Settings(_env_file=None).frame_source == "s3"


def test_frame_source_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAME_SOURCE", "local")
    monkeypatch.setenv("TEMPORAL_API_FRAMES_ROOT", "/data/frames")
    assert Settings(_env_file=None).frame_source == "local"


def test_local_source_without_root_fails_at_startup(monkeypatch):
    # A local-default server without a frames root is dead on every request;
    # fail at boot like other server-level misconfig (model path, threshold).
    monkeypatch.setenv("TEMPORAL_API_FRAME_SOURCE", "local")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_frame_source_rejects_unknown(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAME_SOURCE", "ftp")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_frames_root_default_empty():
    assert Settings(_env_file=None).frames_root == ""


def test_frames_root_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_FRAMES_ROOT", "/data/frames")
    assert Settings(_env_file=None).frames_root == "/data/frames"


def test_api_version_default_none():
    assert Settings(_env_file=None).api_version is None


def test_api_version_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_VERSION", "0.3.0")
    assert Settings(_env_file=None).api_version == "0.3.0"


def test_api_version_empty_env_is_none(monkeypatch):
    # The Dockerfile always sets ENV TEMPORAL_API_VERSION=${VERSION}; a build
    # without the arg yields "" — must normalize to None (not a release).
    monkeypatch.setenv("TEMPORAL_API_VERSION", "")
    assert Settings(_env_file=None).api_version is None
