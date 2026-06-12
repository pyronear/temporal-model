import pytest

import temporal_model.monitor.import_alert_api as _imp_mod
from temporal_model.monitor.import_alert_api import import_alert_api, import_all_orgs
from temporal_model.monitor.store import find_sequence_dirs, read_meta, sequence_exists

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
    def __init__(self, sequences_by_id: dict | None = None):
        self.detection_calls = 0
        self.sequences_by_id: dict[int, dict | None] = sequences_by_id or {}

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

    def get_sequence(self, sequence_id: int) -> dict | None:
        return self.sequences_by_id.get(sequence_id, None)


def fake_download(url: str) -> bytes:
    return b"jpegbytes:" + url.encode()


def test_import_writes_store(tmp_path):
    client = FakeClient()
    stats = import_alert_api(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42307)
    seq_dir = tmp_path / "sis-67" / "donon-sarrebourg-01" / "seq_42307"
    meta = read_meta(seq_dir)
    assert meta.key == "alert-api_42307"
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
    import_alert_api(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    stats = import_alert_api(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert stats == {"imported": 0, "skipped": 1}
    assert client.detection_calls == 1  # second run never re-fetched detections


def test_import_force_redownloads(tmp_path):
    client = FakeClient()
    import_alert_api(
        client, tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    stats = import_alert_api(
        client,
        tmp_path,
        "2026-05-15",
        "2026-05-16",
        force=True,
        download=fake_download,
    )
    assert stats == {"imported": 1, "skipped": 0}
    assert client.detection_calls == 2


def test_import_handles_missing_org_names(tmp_path):
    class NoOrgClient(FakeClient):
        def list_organizations(self):
            raise PermissionError("403")

    import_alert_api(
        NoOrgClient(), tmp_path, "2026-05-15", "2026-05-16", download=fake_download
    )
    assert (tmp_path / "org-11").is_dir()  # falls back to org-<id>


def test_import_creates_store_dir_even_when_empty(tmp_path):
    class EmptyClient(FakeClient):
        def list_sequences_for_date(self, day):
            return []

    store = tmp_path / "store"
    stats = import_alert_api(
        EmptyClient(), store, "2026-01-01", "2026-01-01", download=fake_download
    )
    assert stats == {"imported": 0, "skipped": 0}
    assert store.is_dir()


# ---------------------------------------------------------------------------
# Helpers for scan-mode tests
# ---------------------------------------------------------------------------

# Two cameras in two different orgs.
SCAN_CAMERAS = [
    {"id": 10, "name": "cam-alpha", "organization_id": 1},
    {"id": 20, "name": "cam-beta", "organization_id": 2},
]
SCAN_ORGS = [
    {"id": 1, "name": "sdis-07"},
    {"id": 2, "name": "sdis-77"},
]


def _det(det_id: int, hour: int, letter: str) -> dict:
    return {
        "id": det_id,
        "created_at": f"2026-06-01T{hour:02d}:00:00",
        "bucket_key": f"k/{letter}.jpg",
        "bbox": "",
        "url": f"https://s3.test/{letter}.jpg",
    }


SCAN_DETS = {
    97: [_det(200, 10, "a")],
    99: [_det(201, 11, "b")],
    101: [_det(202, 12, "c")],
    103: [_det(203, 13, "d")],
}


def _seq(sid: int, cam: int, wildfire, started: str) -> dict:
    return {"id": sid, "camera_id": cam, "is_wildfire": wildfire, "started_at": started}


# IDs 97, 99, 101, 103 are in-range; 98, 100, 102 are 404 holes; 104+ are 404.
# ID 95 is before day_from to trigger OLDER_STOP.
_SCAN_SEQS_BY_ID: dict[int, dict | None] = {
    95: _seq(95, 10, None, "2026-05-31T09:00:00"),
    96: None,
    97: _seq(97, 10, "wildfire_smoke", "2026-06-01T10:00:00"),
    98: None,
    99: _seq(99, 20, "other", "2026-06-01T11:00:00"),
    100: None,
    101: _seq(101, 10, None, "2026-06-01T12:00:00"),
    102: None,
    103: _seq(103, 20, "wildfire_smoke", "2026-06-01T13:00:00"),
    # 104, 105, 106 absent → HEAD_GAP stops probing above 103
}


class ScanFakeClient(FakeClient):
    """FakeClient wired for the all-orgs scan tests."""

    def __init__(self, seqs_by_id=None):
        super().__init__(sequences_by_id=seqs_by_id or dict(_SCAN_SEQS_BY_ID))

    def list_sequences_for_date(self, day):
        # own-org listing returns only id=103 (the highest in-range id) for seeding
        if day == "2026-06-01":
            return [self.sequences_by_id[103]]
        return []

    def list_sequence_detections(self, sequence_id):
        self.detection_calls += 1
        return SCAN_DETS.get(sequence_id, [])

    def list_cameras(self):
        return SCAN_CAMERAS

    def list_organizations(self):
        return SCAN_ORGS

    def get_sequence(self, sequence_id: int) -> dict | None:
        return self.sequences_by_id.get(sequence_id, None)


def test_scan_imports_all_orgs(tmp_path, monkeypatch):
    monkeypatch.setattr(_imp_mod, "HEAD_GAP", 3)
    monkeypatch.setattr(_imp_mod, "OLDER_STOP", 3)

    client = ScanFakeClient()
    stats = import_all_orgs(
        client, tmp_path, "2026-06-01", "2026-06-01", download=fake_download
    )
    # IDs 97, 99, 101, 103 are in range across two orgs
    assert stats["imported"] == 4
    assert stats["skipped"] == 0
    # Both orgs' dirs exist
    assert (tmp_path / "sdis-07").is_dir()
    assert (tmp_path / "sdis-77").is_dir()
    # Each in-range sequence was stored
    for sid in (97, 99, 101, 103):
        assert sequence_exists(tmp_path, sid), f"seq_{sid} missing"
    # Head discovery: seed is 103 (from own-org listing); scan walks up to 106
    # (3 consecutive 404s) then back down — just verify ids 104+ don't appear
    for sid in (104, 105, 106):
        assert not sequence_exists(tmp_path, sid)


def test_scan_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(_imp_mod, "HEAD_GAP", 3)
    monkeypatch.setattr(_imp_mod, "OLDER_STOP", 3)

    client = ScanFakeClient()
    import_all_orgs(
        client, tmp_path, "2026-06-01", "2026-06-01", download=fake_download
    )
    stats = import_all_orgs(
        client, tmp_path, "2026-06-01", "2026-06-01", download=fake_download
    )
    assert stats == {"imported": 0, "skipped": 4}
    # No duplicate directories per sequence
    for sid in (97, 99, 101, 103):
        assert len(find_sequence_dirs(tmp_path, sid)) == 1


def test_force_reimport_after_org_rename_leaves_single_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(_imp_mod, "HEAD_GAP", 3)
    monkeypatch.setattr(_imp_mod, "OLDER_STOP", 3)

    client = ScanFakeClient()
    import_all_orgs(
        client, tmp_path, "2026-06-01", "2026-06-01", download=fake_download
    )

    # Rename org 1 from "sdis-07" to "sdis-07-renamed"
    renamed_orgs = [
        {"id": 1, "name": "sdis-07-renamed"},
        {"id": 2, "name": "sdis-77"},
    ]

    class RenamedClient(ScanFakeClient):
        def list_organizations(self):
            return renamed_orgs

    client2 = RenamedClient()
    import_all_orgs(
        client2,
        tmp_path,
        "2026-06-01",
        "2026-06-01",
        force=True,
        download=fake_download,
    )

    # Sequences that belonged to sdis-07 (cam 10: ids 97, 101) should now live
    # under sdis-07-renamed, and the old sdis-07 dir must not hold them.
    for sid in (97, 101):
        dirs = find_sequence_dirs(tmp_path, sid)
        assert len(dirs) == 1, f"seq_{sid} has {len(dirs)} dirs after rename"
        assert "sdis-07-renamed" in str(dirs[0]), f"seq_{sid} not under new org dir"
    # Sequences for org 2 (sdis-77: ids 99, 103) are unchanged
    for sid in (99, 103):
        dirs = find_sequence_dirs(tmp_path, sid)
        assert len(dirs) == 1


def test_scan_seed_error(tmp_path, monkeypatch):
    monkeypatch.setattr(_imp_mod, "HEAD_GAP", 3)
    monkeypatch.setattr(_imp_mod, "OLDER_STOP", 3)

    class EmptyScanClient(ScanFakeClient):
        def list_sequences_for_date(self, day):
            return []

    client = EmptyScanClient(seqs_by_id={})
    with pytest.raises(SystemExit, match="--seed-id"):
        import_all_orgs(
            client, tmp_path, "2026-06-01", "2026-06-01", download=fake_download
        )
