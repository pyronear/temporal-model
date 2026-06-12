from temporal_model.monitor.store import (
    FrameMeta,
    SequenceMeta,
    iter_metas,
    label_from_is_wildfire,
    read_meta,
    sequence_dir,
    sequence_exists,
    slugify,
    write_meta,
)


def make_meta(sequence_id: int = 42307) -> SequenceMeta:
    return SequenceMeta(
        key=f"platform_{sequence_id}",
        sequence_id=sequence_id,
        label="smoke",
        label_detail="wildfire_smoke",
        camera_id=122,
        camera_name="donon-sarrebourg-01",
        organization_id=11,
        organization_name="sis-67",
        started_at="2026-05-15T13:08:18.671072",
        temporal_model_score=0.9867,
        temporal_model_version="0.1.0",
        temporal_api_version="0.3.1",
        frames=[
            FrameMeta(
                file="images/detection_100.jpg",
                detection_id=100,
                created_at="2026-05-15T13:08:18.671072",
                bucket_key="cam122/frame-100.jpg",
                bbox="[(0.1,0.2,0.3,0.4,0.9)]",
            )
        ],
    )


def test_slugify():
    assert slugify("SIS 67") == "sis-67"
    assert slugify("Donon/Sarrebourg_01") == "donon-sarrebourg-01"
    assert slugify("") == "unknown"
    assert slugify(None) == "unknown"


def test_label_from_is_wildfire():
    assert label_from_is_wildfire("wildfire_smoke") == ("smoke", "wildfire_smoke")
    assert label_from_is_wildfire("other_smoke") == ("smoke", "other_smoke")
    assert label_from_is_wildfire("other") == ("fp", "other")
    assert label_from_is_wildfire(None) == ("unknown", None)


def test_meta_roundtrip(tmp_path):
    meta = make_meta()
    seq_dir = sequence_dir(tmp_path, meta)
    write_meta(seq_dir, meta)
    assert seq_dir == tmp_path / "sis-67" / "donon-sarrebourg-01" / "seq_42307"
    assert read_meta(seq_dir) == meta


def test_sequence_exists(tmp_path):
    meta = make_meta()
    assert not sequence_exists(tmp_path, 42307)
    write_meta(sequence_dir(tmp_path, meta), meta)
    assert sequence_exists(tmp_path, 42307)
    assert not sequence_exists(tmp_path, 999)


def test_iter_metas(tmp_path):
    for sid in (1, 2):
        meta = make_meta(sid)
        write_meta(sequence_dir(tmp_path, meta), meta)
    found = list(iter_metas(tmp_path))
    assert sorted(m.sequence_id for _, m in found) == [1, 2]
    assert all(d.name == f"seq_{m.sequence_id}" for d, m in found)
