import threading

from temporal_model.triage.pull import pull_unannotated
from temporal_model.triage.store import (
    SequenceMeta,
    iter_sequence_dirs,
    read_meta,
    sequence_dir,
    sequence_exists,
)


class FakeClient:
    def __init__(self, sequences, detections):
        self._sequences = sequences
        self._detections = detections
        self.detection_calls = 0

    def iter_unannotated_sequences(
        self, *, processing_stage="ready_to_annotate", page_size=100, limit=None
    ):
        self.last_stage = processing_stage
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


def test_pull_forwards_processing_stage(tmp_path):
    client = FakeClient([SEQ], DETS)
    pull_unannotated(client, tmp_path, download=lambda url: b"img")
    assert client.last_stage == "ready_to_annotate"


def test_pull_parallel_downloads_every_frame(tmp_path):
    dets = [
        {"id": i, "recorded_at": f"2026-06-01T10:{i:02d}:00", "bucket_key": f"k/{i}"}
        for i in range(12)
    ]
    seq = dict(SEQ, id=99)
    client = FakeClient([seq], {99: dets})

    seen = set()
    lock = threading.Lock()

    def dl(url):
        with lock:
            seen.add(url)
        return b"img"

    counts = pull_unannotated(client, tmp_path, workers=6, download=dl)
    assert counts == {"pulled": 1, "skipped": 0}
    seq_dir = sequence_dir(
        tmp_path,
        SequenceMeta(
            key="pyro-annotator_99",
            sequence_id=99,
            camera_name="cam-a",
            organization_name="sis-67",
        ),
    )
    files = sorted(p.name for p in (seq_dir / "images").glob("*.jpg"))
    assert len(files) == 12  # all frames written despite concurrency
    assert len(seen) == 12  # each distinct signed URL fetched exactly once


def test_pull_sequence_level_concurrency(tmp_path):
    seqs = [dict(SEQ, id=i, camera_name=f"cam-{i}") for i in range(5)]
    dets = {
        i: [
            {
                "id": i * 10 + j,
                "recorded_at": f"2026-06-01T10:0{j}:00",
                "bucket_key": f"k/{i}-{j}",
            }
            for j in range(3)
        ]
        for i in range(5)
    }
    client = FakeClient(seqs, dets)
    pulled_urls = set()
    lock = threading.Lock()

    def dl(url):
        with lock:
            pulled_urls.add(url)
        return b"img"

    counts = pull_unannotated(
        client, tmp_path, seq_workers=3, workers=2, download=dl
    )
    assert counts == {"pulled": 5, "skipped": 0}
    assert sum(1 for _ in iter_sequence_dirs(tmp_path)) == 5  # all sequences landed
    assert len(pulled_urls) == 15  # 5 sequences x 3 frames, no drops under concurrency


def test_limit_is_forwarded(tmp_path):
    seqs = [dict(SEQ, id=i) for i in (42, 43)]
    dets = {42: DETS[42], 43: DETS[42]}
    client = FakeClient(seqs, dets)
    counts = pull_unannotated(client, tmp_path, limit=1, download=lambda url: b"img")
    assert counts == {"pulled": 1, "skipped": 0}
    assert sequence_exists(tmp_path, 42)
    assert not sequence_exists(tmp_path, 43)
