"""Crop tube bboxes from raw frames into 224x224 PNG patches per tube.

For each tube JSON in ``--tubes-dir``, look up the source sequence
under ``--raw-dir`` and write a directory of cropped PNG frames + a
``meta.json`` to ``--output-dir/<sequence_id>/``. Also writes a
``_index.json`` listing all surviving tubes.

Wipes ``--output-dir`` at the start so stale outputs don't linger.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from temporal_model.core.model_input import LABEL_TO_INT, process_tube


def _process_one(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
) -> tuple[str | None, str, str | None]:
    """Worker: returns (sequence_id, label, error_or_none)."""
    try:
        record = json.loads(tube_path.read_text())
        sequence_id = record["sequence_id"]
        label = record["label"]
        process_tube(
            tube_path=tube_path,
            raw_dir=raw_dir,
            out_dir=out_dir,
            context_factor=context_factor,
            patch_size=patch_size,
        )
        return sequence_id, label, None
    except Exception as exc:  # noqa: BLE001
        return None, "", f"{tube_path.name}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tubes-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-factor", type=float, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = os.cpu_count()).",
    )
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tube_paths = sorted(
        p for p in args.tubes_dir.glob("*.json") if p.name != "_summary.json"
    )

    index: list[dict] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _process_one,
                p,
                args.raw_dir,
                args.output_dir,
                args.context_factor,
                args.patch_size,
            )
            for p in tube_paths
        ]
        for fut in as_completed(futures):
            sequence_id, label, err = fut.result()
            if err is not None:
                errors.append(err)
                continue
            assert sequence_id is not None
            num_frames = len(
                json.loads((args.output_dir / sequence_id / "meta.json").read_text())[
                    "frames"
                ]
            )
            index.append(
                {
                    "sequence_id": sequence_id,
                    "label_int": LABEL_TO_INT[label],
                    "num_frames": num_frames,
                }
            )

    index.sort(key=lambda r: r["sequence_id"])
    (args.output_dir / "_index.json").write_text(json.dumps(index, indent=2))

    split = args.tubes_dir.name
    print(
        f"[{split}] wrote {len(index)}/{len(tube_paths)} tubes "
        f"with {workers} workers (errors={len(errors)})"
    )
    for e in errors[:5]:
        print(f"  error: {e}")
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
