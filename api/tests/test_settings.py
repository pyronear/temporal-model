from temporal_model.api.settings import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model_path == "/models/model.zip"
    assert s.device is None
    assert s.s3_bucket == ""
    assert s.s3_region is None
    assert s.s3_endpoint_url is None
    assert s.port == 8000


def test_env_override(monkeypatch):
    monkeypatch.setenv("TEMPORAL_API_S3_BUCKET", "pyro-frames")
    monkeypatch.setenv("TEMPORAL_API_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("TEMPORAL_API_DEVICE", "cpu")
    s = Settings(_env_file=None)
    assert s.s3_bucket == "pyro-frames"
    assert s.s3_endpoint_url == "http://minio:9000"
    assert s.device == "cpu"
