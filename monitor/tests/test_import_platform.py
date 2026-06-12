from temporal_model.monitor.import_platform import import_platform
from temporal_model.monitor.store import read_meta, sequence_exists

SEQ = {
    "id": 42307,
    "camera_id": 122,
    "is_wildfire": "wildfire_smoke",
    "started_at": "2026-05-15T13:08:18.671072",
    "temporal_model_score": 0.9867,
    "temporal_model_version": "0.1.0",
    "temporal_api_version": "0.3.1",
}
DETS = [
    {
        "id": 100,
        "created_at": "2026-05-15T13:08:18.671072",
        "bucket_key": "cam122/frame-100.jpg",
        "bbox": "[(0.1,0.2,0.3,0.4,0.9)]",
        "url": "https://s3.test/frame-100.jpg?sig=x",
    },
    {
        "id": 101,
        "created_at": "2026-05-15T13:09:18.671072",
        "bucket_key": "cam122/frame-101.jpg",
        "bbox": "[(0.11,0.2,0.31,0.4,0.8)]",
        "url": "https://s3.test/frame-101.jpg?sig=x",
    },
]


class FakeClient:
    def __init__(self):
        self.detection_calls = 0

    def list_sequences_for_date(self, day):
        return [SEQ] if day == "2026-05-15" else []

    def list_sequence_detections(self, sequence_id):
        assert sequence_id == 42307
        self.detection_calls += 1
        return DETS

    def list_cameras(self):
        return [{"id": 122, "name": "donon-sarrebourg-01", "organization_id": 11}]

    def list_organizations(self):
        return [{"id": 11, "name": "sis-67"}]


def fake_download(url: str) -> bytes:
    return b"jpegbytes:" + url.encode()


def test_import_writes_store(tmp_path):
    client = FakeClient()
    stats = import_platform(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42307)
    seq_dir = tmp_path / "sis-67" / "donon-sarrebourg-01" / "seq_42307"
    meta = read_meta(seq_dir)
    assert meta.key == "platform_42307"
    assert meta.label == "smoke"
    assert meta.temporal_api_version == "0.3.1"
    assert [f.bucket_key for f in meta.frames] == [
        "cam122/frame-100.jpg",
        "cam122/frame-101.jpg",
    ]
    assert (
        (seq_dir / "images" / "detection_100.jpg")
        .read_bytes()
        .startswith(b"jpegbytes:")
    )


def test_import_is_incremental(tmp_path):
    client = FakeClient()
    import_platform(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    stats = import_platform(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 0, "skipped": 1}
    assert client.detection_calls == 1  # second run never re-fetched detections


def test_import_handles_missing_org_names(tmp_path):
    class NoOrgClient(FakeClient):
        def list_organizations(self):
            raise PermissionError("403")

    import_platform(
        NoOrgClient(), tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert (tmp_path / "org-11").is_dir()  # falls back to org-<id>
