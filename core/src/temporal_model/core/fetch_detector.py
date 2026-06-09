"""Download the declared companion detector from HuggingFace and verify it.

Reads the detector identity from ``detector.yaml`` (the single source of truth),
downloads its weights file from the HuggingFace repo, asserts the SHA-256 matches
the declared value, and writes the verified weights to an output path (where the
packaging step expects ``yolo_weights.pt``). Reproducible and tamper-evident.

Usage:
    python -m temporal_model.core.fetch_detector --output yolo_weights.pt
"""

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from .detector import DETECTOR_WEIGHTS_FILENAME, Detector, load_detector

__all__ = ["fetch_detector", "main"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_detector(output_path: Path, detector: Detector | None = None) -> Path:
    """Download the detector weights, verify the SHA-256, write to ``output_path``.

    Raises:
        ValueError: if the downloaded weights' SHA-256 does not match the
            declared ``detector.sha256``.
    """
    det = detector or load_detector()
    downloaded = Path(
        hf_hub_download(repo_id=det.repo_id, filename=DETECTOR_WEIGHTS_FILENAME)
    )
    actual = _sha256(downloaded)
    if actual != det.sha256:
        raise ValueError(
            f"Detector SHA-256 mismatch for {det.name}: "
            f"expected {det.sha256}, got {actual}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the verified detector weights (e.g. yolo_weights.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    det = load_detector()
    out = fetch_detector(args.output, det)
    print(f"Fetched {det.name} -> {out} (sha256 {det.sha256} verified)")


if __name__ == "__main__":
    main()
