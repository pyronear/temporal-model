from temporal_model.triage.pull import pull_unannotated
from temporal_model.triage.store import (
    SequenceMeta,
    read_meta,
    sequence_dir,
    sequence_exists,
)


class FakeClient:
    def __init__(self, sequences, detections):
        self._sequences = sequences
        self._detections = detections
        self.detection_calls = 0

    def iter_unannotated_sequences(self, *, page_size=100, limit=None):
        seqs = self._sequences if limit is None else self._sequences[:limit]
        yield from seqs

    def list_detections(self, sequence_id):
        self.detection_calls += 1
        return self._detections[sequence_id]

    def detection_image_url(self, detection_id):
        return f"https://s3.test/det-{detection_id}.jpg?sig=x"


SEQ = {
    "id": 42,
    "camera_id": 1,
    "camera_name": "cam-a",
    "organisation_id": 9,
    "organisation_name": "sis-67",
    "recorded_at": "2026-06-01T10:00:00",
}
DETS = {
    42: [
        {"id": 7, "recorded_at": "2026-06-01T10:00:00", "bucket_key": "k/7.jpg"},
        {"id": 8, "recorded_at": "2026-06-01T10:01:00", "bucket_key": "k/8.jpg"},
    ]
}


def _expected_meta() -> SequenceMeta:
    return SequenceMeta(
        key="pyro-annotator_42",
        sequence_id=42,
        camera_name="cam-a",
        organization_name="sis-67",
    )


def test_pull_writes_store_and_downloads_frames(tmp_path):
    client = FakeClient([SEQ], DETS)
    downloads = []
    counts = pull_unannotated(
        client, tmp_path, download=lambda url: downloads.append(url) or b"img"
    )
    assert counts == {"pulled": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42)
    seq_dir = sequence_dir(tmp_path, _expected_meta())
    meta = read_meta(seq_dir)
    assert meta.key == "pyro-annotator_42"
    assert [f.detection_id for f in meta.frames] == [7, 8]
    assert (seq_dir / "images/detection_7.jpg").read_bytes() == b"img"
    assert len(downloads) == 2


def test_pull_is_incremental(tmp_path):
    client = FakeClient([SEQ], DETS)
    pull_unannotated(client, tmp_path, download=lambda url: b"img")
    counts = pull_unannotated(client, tmp_path, download=lambda url: b"img")
    assert counts == {"pulled": 0, "skipped": 1}


def test_limit_is_forwarded(tmp_path):
    seqs = [dict(SEQ, id=i) for i in (42, 43)]
    dets = {42: DETS[42], 43: DETS[42]}
    client = FakeClient(seqs, dets)
    counts = pull_unannotated(client, tmp_path, limit=1, download=lambda url: b"img")
    assert counts == {"pulled": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42)
    assert not sequence_exists(tmp_path, 43)
