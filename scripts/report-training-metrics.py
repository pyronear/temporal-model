#!/usr/bin/env python3
"""Render the training-result PR body from eval metrics snapshots.

Reads the metrics.json snapshots the train workflow copies into results/
(one per eval source) and, when available, the baseline snapshots committed
on main. Prints a Markdown PR body on stdout.

The pyro-annotator source is the fixed testbed, so it gets the full
old-vs-new comparison; train/val change with every dataset release, so they
are reported as-is (calibration check only, see the retrain runbook).
"""

import argparse
import json
import sys
from pathlib import Path

SOURCES = ("pyro-annotator", "train", "val")

# (label, metrics.json key, is_ratio) — ratios get 4 decimals, counts don't.
PYRO_ANNOTATOR_ROWS = [
    ("False alerts (FP)", "fp", False),
    ("Missed smoke (FN)", "fn", False),
    ("Precision", "precision", True),
    ("Recall", "recall", True),
    ("F1", "f1", True),
    ("FPR", "fpr", True),
    ("Median TTD (frames)", "median_ttd_frames", False),
    ("Mean TTD (frames)", "mean_ttd_frames", False),
    ("ROC AUC", "roc_auc", True),
    ("PR AUC", "pr_auc", True),
]

SPLIT_COLUMNS = [
    ("n", "num_sequences", False),
    ("precision", "precision", True),
    ("recall", "recall", True),
    ("F1", "f1", True),
    ("FPR", "fpr", True),
    ("ROC AUC", "roc_auc", True),
]


def fmt(value, is_ratio: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if is_ratio else f"{value:g}"


def delta(new, old, is_ratio: bool) -> str:
    if new is None or old is None:
        return "n/a"
    d = new - old
    return f"{d:+.4f}" if is_ratio else f"{d:+g}"


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def pyro_annotator_section(curr: dict, base: dict | None) -> str:
    lines = [
        "## pyro-annotator (fixed testbed, old vs new)",
        "",
        f"Sequences: {curr.get('num_sequences', 'n/a')}"
        + ("" if base is None else " — baseline = results/ on main"),
        "",
        "| Metric | main | current | Δ |",
        "|--------|------|---------|---|",
    ]
    for label, key, is_ratio in PYRO_ANNOTATOR_ROWS:
        old = base.get(key) if base else None
        new = curr.get(key)
        lines.append(
            f"| {label} | {fmt(old, is_ratio)} | {fmt(new, is_ratio)} "
            f"| {delta(new, old, is_ratio)} |"
        )
    if base is None:
        lines += ["", "_(no baseline on main yet)_"]
    return "\n".join(lines)


def splits_section(train: dict | None, val: dict | None) -> str:
    header = " | ".join(label for label, _, _ in SPLIT_COLUMNS)
    sep = "|".join("---" for _ in SPLIT_COLUMNS)
    lines = [
        "## train / val (dataset-dependent, calibration check only)",
        "",
        f"| Source | {header} |",
        f"|--------|{sep}|",
    ]
    for name, metrics in (("train (in-sample)", train), ("val", val)):
        if metrics is None:
            continue
        cells = " | ".join(fmt(metrics.get(key), r) for _, key, r in SPLIT_COLUMNS)
        lines.append(f"| {name} | {cells} |")
    lines += [
        "",
        "Val recall should sit at the calibration target "
        "(`package.target_recall` in train/params.yaml); old-vs-new deltas "
        "are not meaningful here when the dataset changed.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dataset-rev", default="unchanged")
    parser.add_argument("--result-branch", default="")
    args = parser.parse_args()

    current: dict[str, dict] = {}
    for src in SOURCES:
        metrics = load(args.results_dir / f"{src}.json")
        if metrics is None:
            path = args.results_dir / f"{src}.json"
            print(f"missing metrics snapshot: {path}", file=sys.stderr)
            return 1
        current[src] = metrics
    baseline = {src: load(args.baseline_dir / f"{src}.json") for src in SOURCES}

    sections = [
        pyro_annotator_section(current["pyro-annotator"], baseline["pyro-annotator"]),
        splits_section(current["train"], current["val"]),
        f"**Branch:** `{args.result_branch}` | **Dataset rev:** `{args.dataset_rev}`",
    ]
    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
