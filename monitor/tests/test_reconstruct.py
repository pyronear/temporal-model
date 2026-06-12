from temporal_model.monitor.reconstruct import (
    MAX_FRAMES,
    MIN_FRAMES,
    extract_bbox_strings,
    frames_and_roi,
    parse_bbox,
)
from temporal_model.monitor.store import FrameMeta


def frame(i: int, bucket_key: str, bbox: str) -> FrameMeta:
    return FrameMeta(
        file=f"images/detection_{i}.jpg",
        detection_id=i,
        created_at=f"2026-05-15T13:{i:02d}:00",
        bucket_key=bucket_key,
        bbox=bbox,
    )


def test_constants_match_pyro_api():
    assert MIN_FRAMES == 4
    assert MAX_FRAMES == 10


def test_extract_and_parse_bbox():
    strs = extract_bbox_strings("[(0.1,0.2,0.3,0.4,0.9),(0.5,0.5,0.6,0.6,0.2)]")
    assert strs == ["(0.1,0.2,0.3,0.4,0.9)", "(0.5,0.5,0.6,0.6,0.2)"]
    assert parse_bbox(strs[0]) == (0.1, 0.2, 0.3, 0.4, 0.9)
    assert parse_bbox("garbage") is None
    assert parse_bbox("(0.1,0.2)") is None


def test_frames_distinct_and_ordered():
    frames = [
        frame(0, "k0.jpg", "[(0.1,0.1,0.2,0.2,0.9)]"),
        frame(1, "k0.jpg", "[(0.15,0.1,0.25,0.2,0.9)]"),  # same frame, 2nd detection
        frame(2, "k1.jpg", "[(0.2,0.1,0.3,0.2,0.9)]"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert total == 2
    assert kept == ["k0.jpg", "k1.jpg"]
    # envelope of the three primary bboxes
    assert roi == [0.1, 0.1, 0.3, 0.2]


def test_truncates_to_last_max_frames_and_roi_covers_kept_only():
    frames = [
        frame(i, f"k{i}.jpg", f"[(0.{i},0.1,0.{i + 1},0.2,0.9)]") for i in range(1, 9)
    ] + [
        frame(20, "k20.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
        frame(21, "k21.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
        frame(22, "k22.jpg", "[(0.5,0.5,0.6,0.6,0.9)]"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert total == 11
    assert len(kept) == MAX_FRAMES
    assert kept[0] == "k2.jpg"  # k1 truncated away
    # k1's bbox (xmin 0.1) is truncated away; the kept envelope starts at k2's 0.2
    assert roi[0] == 0.2


def test_roi_none_when_no_bbox_parses():
    frames = [frame(i, f"k{i}.jpg", "") for i in range(5)]
    total, kept, roi = frames_and_roi(frames)
    assert total == 5
    assert roi is None


def test_roi_none_when_degenerate():
    # a single zero-area box -> xmin == xmax -> degenerate -> None
    frames = [frame(0, "k0.jpg", "[(0.5,0.5,0.5,0.5,0.9)]")]
    _, _, roi = frames_and_roi(frames)
    assert roi is None


def test_roi_clamped_to_unit_square():
    # parseable floats are already in [0,1] per the regex, but clamping is kept
    # for parity with pyro-api (max(0,...), min(1,...))
    frames = [frame(0, "k0.jpg", "[(0,0,1,1,0.9)]")]
    _, _, roi = frames_and_roi(frames)
    assert roi == [0.0, 0.0, 1.0, 1.0]


def test_unparseable_primary_bbox_skipped_not_fatal():
    frames = [
        frame(0, "k0.jpg", "[(0.1,0.1,0.2,0.2,0.9)]"),
        frame(1, "k1.jpg", "not-a-bbox"),
    ]
    total, kept, roi = frames_and_roi(frames)
    assert kept == ["k0.jpg", "k1.jpg"]
    assert roi == [0.1, 0.1, 0.2, 0.2]
