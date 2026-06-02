"""Evaluation entry point.

Scaffold stub — the real DVC-driven evaluation is migrated later.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the temporal smoke classifier."
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.parse_args()
    raise SystemExit("temporal-eval: not implemented yet (scaffold stub)")


if __name__ == "__main__":
    main()
