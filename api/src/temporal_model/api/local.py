"""Local-filesystem frame resolution.

Resolves request frame paths (relative identifiers) under the
server-configured ``frames_root`` and returns them in request order. Unlike
the S3 path, nothing is copied: the returned paths point at the real files.
The root comes from settings only — never from the request — so a request
cannot reference paths outside it (see
docs/specs/2026-06-11-api-local-frames-design.md, decision 2).
"""

from pathlib import Path

from .errors import FrameNotFound, InvalidRequest


def resolve_frames(root: Path, frames: list[str]) -> list[Path]:
    """Resolve ``frames`` under ``root``, in request order.

    Rejects absolute paths and ``..`` segments outright, and anything whose
    resolved path (symlinks followed) lands outside ``root``. A missing file
    raises :class:`FrameNotFound` — the same error a missing S3 key maps to.
    Error messages echo the request string, never the resolved server path.
    """
    root = root.resolve()
    if not root.is_dir():
        # A typo'd or unmounted root would otherwise surface as a 404 per
        # frame, indistinguishable from genuinely missing frames (the local
        # analog of fetch_frames mapping NoSuchBucket to a distinct error).
        raise InvalidRequest(
            "frames root is not a directory: check TEMPORAL_API_FRAMES_ROOT"
        )
    paths: list[Path] = []
    for frame in frames:
        rel = Path(frame)
        # Empty/"." have no parts and would resolve to the root itself.
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise InvalidRequest(
                f"frame must be a non-empty relative path without '..': {frame!r}"
            )
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root):
            raise InvalidRequest(f"frame escapes the frames root: {frame!r}")
        if not resolved.is_file():
            raise FrameNotFound(f"frame not found: {frame}")
        paths.append(resolved)
    return paths
