"""Training entry point.

Scaffold stub — the real DVC-driven training loop is migrated later.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal smoke classifier.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.parse_args()
    raise SystemExit("temporal-train: not implemented yet (scaffold stub)")


if __name__ == "__main__":
    main()
