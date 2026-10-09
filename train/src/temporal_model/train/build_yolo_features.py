"""Precompute YOLO neck embeddings for every tube entry (YOLO-feature variants).

For each tube JSON in ``--tubes-dir``, run the companion detector over the
tube's frames and ROI-pool P3/P4/P5 at each entry's box (plus P5 whole-frame
context), via the same :class:`YoloFeatureExtractor` the inference path uses.
Writes ``--output-dir/<sequence_id>.npz`` with float16 arrays ``p3``, ``p4``,
``p5``, ``ctx`` — one row per tube entry, aligned with the patch slots that
``build_model_input`` writes. All levels are stored; variants pick theirs.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from temporal_model.core.package import load_yolo
from temporal_model.core.sequences import find_sequence_dir
from temporal_model.core.yolo_features import YoloFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tubes-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = YoloFeatureExtractor(
        load_yolo(args.yolo_weights), image_size=args.image_size, device=device
    )
    tube_paths = sorted(
        p for p in args.tubes_dir.glob("*.json") if p.name != "_summary.json"
    )
    for i, path in enumerate(tube_paths, start=1):
        record = json.loads(path.read_text())
        seq_dir = find_sequence_dir(args.raw_dir, record["sequence_id"])
        if seq_dir is None:
            raise FileNotFoundError(f"raw sequence dir not found for {path.name}")
        entries = [
            (seq_dir / "images" / f"{e['frame_id']}.jpg", tuple(e["bbox"]))
            for e in record["tube"]["entries"]
        ]
        feats = extractor.tube_features(entries)
        np.savez(
            args.output_dir / f"{record['sequence_id']}.npz",
            **{k: v.astype(np.float16) for k, v in feats.items()},
        )
        if i % 200 == 0:
            print(f"[{args.tubes_dir.name}] {i}/{len(tube_paths)} tubes", flush=True)
    print(f"[{args.tubes_dir.name}] wrote {len(tube_paths)} feature files ({device})")


if __name__ == "__main__":
    main()
