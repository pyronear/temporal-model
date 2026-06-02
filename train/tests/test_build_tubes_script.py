"""Unit tests for the build_tubes data-prep module's per-sequence processing."""

from pathlib import Path

import pytest

from temporal_model.train import build_tubes as build_tubes_module


@pytest.fixture(scope="module")
def script_module():
    return build_tubes_module


def _write_wf_sequence(root: Path, name: str, lines_per_frame: list[list[str]]) -> Path:
    """Create a tiny 'wildfire' GT sequence under root/wildfire/<name>/.

    Writes a 0-byte image stub per frame (get_sorted_frames globs ``images/*.jpg``
    to drive ordering; the loader doesn't actually open the files) plus the
    matching label file.
    """
    seq = root / "wildfire" / name
    (seq / "labels").mkdir(parents=True)
    (seq / "images").mkdir(parents=True)
    for i, lines in enumerate(lines_per_frame):
        stem = f"{name}_2024-01-01T00-00-{i:02d}"
        (seq / "images" / f"{stem}.jpg").write_bytes(b"")
        (seq / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
    return seq


def test_merge_keys_fuse_a_growing_plume(tmp_path, script_module):
    """Two frames of a tiny box then two of a big box -> the merge unites them
    into one tube longer than what the no-merge path would select.
    """
    # 5-col GT format: "class cx cy w h" (no confidence)
    seq = _write_wf_sequence(
        tmp_path,
        "seq01",
        lines_per_frame=[
            ["0 0.5 0.5 0.02 0.02"],
            ["0 0.5 0.5 0.02 0.02"],
            ["0 0.5 0.5 0.20 0.20"],
            ["0 0.5 0.5 0.20 0.20"],
        ],
    )
    record_with, _ = script_module._process_sequence(
        seq,
        split="train",
        iou_threshold=0.2,
        max_misses=2,
        min_tube_length=2,
        min_detected_entries=2,
        merge_iomin=0.3,
        merge_prox_factor=1.0,
        merge_max_gap=10,
    )
    record_without, _ = script_module._process_sequence(
        seq,
        split="train",
        iou_threshold=0.2,
        max_misses=2,
        min_tube_length=2,
        min_detected_entries=2,
        merge_iomin=None,
        merge_prox_factor=None,
        merge_max_gap=None,
    )
    assert record_with is not None and record_without is not None
    with_tube = record_with["tube"]
    without_tube = record_without["tube"]
    # Merge yields a single tube spanning all 4 frames; legacy picks one of the
    # two fragments (length 2).
    assert with_tube["end_frame"] - with_tube["start_frame"] + 1 == 4
    assert without_tube["end_frame"] - without_tube["start_frame"] + 1 == 2
