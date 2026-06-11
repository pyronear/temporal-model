from pathlib import Path

import pytest

from temporal_model.api.errors import FrameNotFound, InvalidRequest
from temporal_model.api.local import resolve_frames

FRAMES = [
    "cam12/2023-05-23/adf_2023-05-23T17-18-01.jpg",
    "cam12/2023-05-23/adf_2023-05-23T17-18-31.jpg",
]


def _make_frames(root: Path, frames=FRAMES) -> None:
    for frame in frames:
        p = root / frame
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")


def test_resolve_preserves_order_and_points_at_real_files(tmp_path):
    _make_frames(tmp_path)
    paths = resolve_frames(tmp_path, FRAMES)
    # The real files, in request order — no copy.
    assert paths == [(tmp_path / f).resolve() for f in FRAMES]
    assert all(p.read_bytes() == b"\xff\xd8\xff\xe0jpegbytes" for p in paths)


def test_missing_file_raises_frame_not_found(tmp_path):
    with pytest.raises(FrameNotFound):
        resolve_frames(tmp_path, ["cam12/missing.jpg"])


def test_directory_raises_frame_not_found(tmp_path):
    (tmp_path / "cam12").mkdir()
    with pytest.raises(FrameNotFound):
        resolve_frames(tmp_path, ["cam12"])


def test_absolute_path_raises_invalid_request(tmp_path):
    _make_frames(tmp_path)
    inside = tmp_path / FRAMES[0]
    # Even an absolute path pointing inside the root is rejected: the contract
    # is relative identifiers only.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, [str(inside)])


def test_dotdot_raises_invalid_request_even_when_inside(tmp_path):
    _make_frames(tmp_path)
    # Resolves inside the root, but `..` segments are rejected outright.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, [f"cam12/../{FRAMES[0]}"])


def test_escaping_dotdot_raises_invalid_request(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.jpg").write_bytes(b"x")
    with pytest.raises(InvalidRequest):
        resolve_frames(root, ["../secret.jpg"])


def test_symlink_escaping_root_raises_invalid_request(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    (root / "link.jpg").symlink_to(outside)
    with pytest.raises(InvalidRequest):
        resolve_frames(root, ["link.jpg"])


def test_empty_frame_raises_invalid_request(tmp_path):
    # Path("").parts == () slips past the absolute/".." guards and would
    # resolve to the root itself — a malformed request, not a missing frame.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, [""])


def test_dot_frame_raises_invalid_request(tmp_path):
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path, ["."])


def test_missing_root_raises_invalid_request(tmp_path):
    # A typo'd/unmounted frames root must be a distinct config error, not a
    # per-frame 404 indistinguishable from genuinely missing frames.
    with pytest.raises(InvalidRequest):
        resolve_frames(tmp_path / "nope", FRAMES)


def test_file_root_raises_invalid_request(tmp_path):
    f = tmp_path / "root.txt"
    f.write_bytes(b"x")
    with pytest.raises(InvalidRequest):
        resolve_frames(f, FRAMES)


def test_error_message_echoes_request_string_not_server_path(tmp_path):
    with pytest.raises(FrameNotFound) as exc_info:
        resolve_frames(tmp_path, ["cam12/missing.jpg"])
    assert "cam12/missing.jpg" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
