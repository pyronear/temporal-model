"""Sequence-directory discovery for the nested pyro-dataset layout."""

from pathlib import Path

__all__ = [
    "list_sequences",
    "find_sequence_dir",
    "is_wf_sequence",
    "get_sorted_frames",
]


def list_sequences(split_dir: Path) -> list[Path]:
    """List all sequence directories in a split, sorted by name.

    Supports the nested pyro-dataset v3.0.0 layout::

        split_dir/{wildfire,fp}/<sequence>/

    Returns:
        Sorted list of sequence directory paths.
    """
    sequences: list[Path] = []
    for category in ("wildfire", "fp"):
        cat_dir = split_dir / category
        if cat_dir.is_dir():
            sequences.extend(d for d in sorted(cat_dir.iterdir()) if d.is_dir())
    sequences.sort(key=lambda p: p.name)
    return sequences


def find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    """Find a sequence directory by ID within the nested layout."""
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def is_wf_sequence(sequence_dir: Path) -> bool:
    """Determine if a sequence is wildfire based on parent directory name."""
    return sequence_dir.parent.name == "wildfire"


def get_sorted_frames(sequence_dir: Path) -> list[Path]:
    """Return image paths from a sequence directory sorted by timestamp.

    Looks for ``*.jpg`` files in ``sequence_dir/images/``.
    """
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"), key=lambda p: p.stem)
