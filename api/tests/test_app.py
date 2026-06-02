from fastapi.testclient import TestClient

from temporal_model.api.app import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_stub_returns_501():
    response = client.post("/predict", json={"frame_paths": []})
    assert response.status_code == 501
