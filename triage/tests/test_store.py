from pathlib import Path

from temporal_model.triage.store import (
    IMAGES_DIR,
    FrameRef,
    SequenceMeta,
    build_frames,
    iter_sequence_dirs,
    read_meta,
    sequence_dir,
    sequence_exists,
    slugify,
    write_meta,
)


def _meta() -> SequenceMeta:
    return SequenceMeta(
        key="pyro-annotator_42",
        sequence_id=42,
        camera_name="Donon / Sarrebourg 01",
        organization_name="SIS 67",
        started_at="2026-06-01T10:00:00",
        frames=[
            FrameRef(
                file=f"{IMAGES_DIR}/detection_7.jpg",
                detection_id=7,
                recorded_at="2026-06-01T10:00:00",
                bucket_key="cam1/frame-7.jpg",
            )
        ],
    )


def test_slugify_is_filesystem_safe():
    assert slugify("SIS 67 / Est") == "sis-67-est"
    assert slugify(None) == "unknown"


def test_sequence_dir_layout():
    meta = _meta()
    d = sequence_dir(Path("/store"), meta)
    assert d == Path("/store/sis-67/donon-sarrebourg-01/seq_42")


def test_write_then_read_roundtrip(tmp_path):
    meta = _meta()
    seq_dir = sequence_dir(tmp_path, meta)
    write_meta(seq_dir, meta)
    assert sequence_exists(tmp_path, 42)
    back = read_meta(seq_dir)
    assert back.key == "pyro-annotator_42"
    assert back.label == "unknown"
    assert back.frames[0].detection_id == 7
    assert [d for d in iter_sequence_dirs(tmp_path)] == [seq_dir]


def test_build_frames_orders_and_parses_timestamps(tmp_path):
    meta = _meta()
    seq_dir = sequence_dir(tmp_path, meta)
    frames = build_frames(seq_dir, meta)
    assert len(frames) == 1
    assert frames[0].frame_id == "detection_7"
    assert frames[0].image_path == seq_dir / "images/detection_7.jpg"
    assert frames[0].timestamp is not None
