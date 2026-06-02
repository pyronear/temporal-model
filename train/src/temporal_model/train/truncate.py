"""Truncate sequences to a maximum number of frames.

Copies sequences from an input directory to an output directory,
keeping only the first N frames (sorted by filename/timestamp).
Preserves the {wildfire,fp}/<sequence>/{images,labels}/ structure.
"""

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input split directory (e.g. data/01_raw/datasets_full/train)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output split directory (e.g. data/01_raw/datasets/train)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of frames per sequence (default: 20)",
    )
    return parser.parse_args()


def truncate_sequence(src_seq: Path, dst_seq: Path, max_frames: int) -> None:
    """Copy a sequence directory, truncating to max_frames."""
    src_images = sorted((src_seq / "images").glob("*.jpg"))
    src_labels = src_seq / "labels"

    # Keep first max_frames (earliest by filename/timestamp)
    kept_images = src_images[:max_frames]
    kept_stems = {p.stem for p in kept_images}

    # Copy images
    dst_images = dst_seq / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    for img in kept_images:
        shutil.copy2(img, dst_images / img.name)

    # Copy matching labels
    dst_labels = dst_seq / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)
    if src_labels.is_dir():
        for label_file in sorted(src_labels.glob("*.txt")):
            if label_file.stem in kept_stems:
                shutil.copy2(label_file, dst_labels / label_file.name)


def main() -> None:
    args = parse_args()

    for category in ("wildfire", "fp"):
        src_cat = args.input_dir / category
        if not src_cat.is_dir():
            print(f"Skipping {category}: {src_cat} not found")
            continue

        sequences = sorted(d for d in src_cat.iterdir() if d.is_dir())
        print(f"{category}: {len(sequences)} sequences")

        for seq_dir in sequences:
            dst_seq = args.output_dir / category / seq_dir.name
            if dst_seq.exists():
                continue
            truncate_sequence(seq_dir, dst_seq, args.max_frames)

    print("Done.")


if __name__ == "__main__":
    main()
