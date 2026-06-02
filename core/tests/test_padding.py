"""Tests for pad_frames_* helpers.

The helpers return ``(padded_frames, padded_indices)``. ``padded_indices``
lists the slots in ``padded_frames`` that are synthesised duplicates (never
the real frames). Tests document the index semantics used by
``BboxTubeDetails.preprocessing.padded_frame_indices``.
"""

from pathlib import Path

from temporal_model.core.inference import (
    pad_frames_symmetrically,
    pad_frames_uniform,
)
from temporal_model.core.protocol import Frame


def _mk_frames(n: int) -> list[Frame]:
    return [
        Frame(frame_id=f"f{i}", image_path=Path(f"/tmp/f{i}.jpg")) for i in range(n)
    ]


def test_pad_symmetrically_returns_padded_indices() -> None:
    frames = _mk_frames(2)  # A, B -> [A, A, B] -> [A, A, B, B] -> [A, A, A, B, B]
    padded, indices = pad_frames_symmetrically(frames, min_length=5)
    assert len(padded) == 5
    # After three pad steps (prepend, append, prepend), real frames at slots 2, 3.
    assert indices == [0, 1, 4]


def test_pad_symmetrically_noop_returns_empty_indices() -> None:
    frames = _mk_frames(6)
    padded, indices = pad_frames_symmetrically(frames, min_length=3)
    assert padded == frames
    assert indices == []


def test_pad_symmetrically_empty_input_returns_empty_indices() -> None:
    padded, indices = pad_frames_symmetrically([], min_length=3)
    assert padded == []
    assert indices == []


def test_pad_uniform_returns_padded_indices() -> None:
    frames = _mk_frames(2)
    padded, indices = pad_frames_uniform(frames, min_length=6)
    # Source mapping i*2//6 for i in 0..5 = [0,0,0,1,1,1]
    # Real frames are the first occurrences (slots 0 and 3); all others are padded.
    assert len(padded) == 6
    assert indices == [1, 2, 4, 5]


def test_pad_uniform_noop_returns_empty_indices() -> None:
    frames = _mk_frames(6)
    padded, indices = pad_frames_uniform(frames, min_length=3)
    assert padded == frames
    assert indices == []
