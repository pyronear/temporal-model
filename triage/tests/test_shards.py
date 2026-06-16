import json

from temporal_model.triage.shards import AGGREGATE_FILES, pack, unpack
from temporal_model.triage.store import (
    IMAGES_DIR,
    FrameRef,
    SequenceMeta,
    sequence_dir,
    write_meta,
)


def _meta(i: int) -> SequenceMeta:
    return SequenceMeta(
        key=f"pyro-annotator_{i}",
        sequence_id=i,
        camera_name=f"cam{i}",
        organization_name="org",
        frames=[FrameRef(file=f"{IMAGES_DIR}/detection_{i}.jpg", detection_id=i)],
    )


def _add_sequence(store, report, i: int) -> SequenceMeta:
    meta = _meta(i)
    sd = sequence_dir(store, meta)
    (sd / IMAGES_DIR).mkdir(parents=True)
    (sd / IMAGES_DIR / f"detection_{i}.jpg").write_bytes(b"img" * 100)
    write_meta(sd, meta)
    (report / "details" / f"{meta.key}.json").write_text(
        json.dumps({"tubes": {"kept": []}})
    )
    (report / "sequences" / f"{meta.key}.json").write_text(
        json.dumps({"key": meta.key})
    )
    return meta


def _make(root):
    store, report = root / "store", root / "report"
    (report / "details").mkdir(parents=True)
    (report / "sequences").mkdir(parents=True)
    for i in (1, 2, 3):
        _add_sequence(store, report, i)
    for agg in AGGREGATE_FILES:
        (report / agg).write_bytes(b"AGG")
    return store, report


def test_pack_then_unpack_roundtrip(tmp_path):
    store, report = _make(tmp_path)
    shards = tmp_path / "shards"
    pack(store, report, shards, target_bytes=500)  # tiny → forces multiple shards

    assert (shards / "frames" / "manifest.json").exists()
    assert len(list((shards / "frames").glob("shard_*.tar"))) >= 2
    assert len(list((shards / "report").glob("shard_*.tar"))) >= 1
    assert (shards / "results.parquet").read_bytes() == b"AGG"

    store2, report2 = tmp_path / "store2", tmp_path / "report2"
    unpack(shards, store2, report2)

    sd = sequence_dir(store2, _meta(1))
    assert (sd / "meta.json").exists()
    assert (sd / "images" / "detection_1.jpg").read_bytes() == b"img" * 100
    assert (report2 / "details" / "pyro-annotator_1.json").exists()
    assert (report2 / "sequences" / "pyro-annotator_2.json").exists()
    assert (report2 / "results.parquet").read_bytes() == b"AGG"


def test_pack_frames_is_append_only(tmp_path):
    store, report = _make(tmp_path)
    shards = tmp_path / "shards"
    pack(store, report, shards, target_bytes=10**9)
    m1 = json.loads((shards / "frames" / "manifest.json").read_text())
    tars1 = {p.name for p in (shards / "frames").glob("shard_*.tar")}

    _add_sequence(store, report, 4)  # a new sequence arrives
    pack(store, report, shards, target_bytes=10**9)
    m2 = json.loads((shards / "frames" / "manifest.json").read_text())

    assert set(m1["items"]) == {"1", "2", "3"}
    assert set(m2["items"]) == {"1", "2", "3", "4"}
    # existing sequences keep their original shard (append-only, not rewritten)
    assert m2["items"]["1"] == m1["items"]["1"]
    assert tars1 <= {p.name for p in (shards / "frames").glob("shard_*.tar")}
