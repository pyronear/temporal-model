"""One-time copy of the explorer's enriched pyro-annotator sequences into eval.

Copies frames + meta.json from the temporal-model-explorer store into eval's
data/01_raw/pyro-annotator/. Run once, then `dvc add` the result so it travels
via the eval DVC remote. Requires the explorer checkout to be present locally
with its data pulled (`dvc pull` in the explorer).
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_SRC = Path(
    "../../vision-rd/experiments/temporal-models/temporal-model-explorer/"
    "data/03_primary/sequences/pyro-annotator"
)
DEFAULT_DST = Path("data/01_raw/pyro-annotator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src} (is the explorer data pulled?)")
    args.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for meta in args.src.rglob("meta.json"):
        rel = meta.parent.relative_to(args.src)
        shutil.copytree(meta.parent, args.dst / rel, dirs_exist_ok=True)
        n += 1
    print(f"copied {n} sequences -> {args.dst}")


if __name__ == "__main__":
    main()
