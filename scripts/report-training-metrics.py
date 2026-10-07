#!/usr/bin/env python3
"""Render the training-result PR body from two evaluations of the same test set.

The train workflow scores the new model and main's model on pyro-dataset's
sequential_test (the one set models are compared on) and passes both eval
reporting directories here: each holds metrics.json and predictions.json.
Re-scoring main's model in the same run, with main's code, is what keeps the
comparison valid when the test set grows or the inference code changes.
Prints a Markdown PR body.

Metrics are the prevalence-free ones (recall, FPR, detection delay over all
smoke): the test set's smoke/FP ratio is arbitrary, so precision is not shown.
The paired section counts the sequences the two models decide differently and
gives McNemar's exact p-value, separately for smoke and for false positives.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# (label, metrics.json key, is_ratio, ci key) — ratios get 4 decimals.
ROWS = [
    ("Recall", "recall", True, "recall_ci95"),
    ("FPR", "fpr", True, "fpr_ci95"),
    ("Missed smoke (FN)", "fn", False, None),
    ("False alerts (FP)", "fp", False, None),
    ("Detected within 2 frames", ("detected_within_frames", "2"), True, None),
    ("Detected within 3 frames", ("detected_within_frames", "3"), True, None),
    ("Detected within 5 frames", ("detected_within_frames", "5"), True, None),
    ("FPR @ recall 0.95 (test sweep)", "fpr_at_recall_95", True, None),
    ("ROC AUC", "roc_auc", True, None),
    ("Median TTD, detected only (frames)", "median_ttd_frames", False, None),
]


def get(metrics: dict, key):
    if isinstance(key, tuple):
        return (metrics.get(key[0]) or {}).get(key[1])
    return metrics.get(key)


def fmt(value, is_ratio: bool, ci=None) -> str:
    if value is None:
        return "n/a"
    out = f"{value:.4f}" if is_ratio else f"{value:g}"
    return out if ci is None else f"{out} [{ci[0]:.3f}, {ci[1]:.3f}]"


def delta(new, old, is_ratio: bool) -> str:
    if new is None or old is None:
        return "n/a"
    d = new - old
    return f"{d:+.4f}" if is_ratio else f"{d:+g}"


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_eval(eval_dir: Path) -> tuple[dict, dict[str, dict]]:
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    by_id: dict[str, dict] = {}
    for p in json.loads((eval_dir / "predictions.json").read_text()):
        if p["sequence_id"] in by_id:
            raise SystemExit(f"{eval_dir}: duplicate sequence {p['sequence_id']}")
        by_id[p["sequence_id"]] = p
    return metrics, by_id


def paired(current: dict[str, dict], baseline: dict[str, dict]) -> dict[str, tuple]:
    """Per label: (sequences the new model fixed, sequences it broke)."""
    if current.keys() != baseline.keys():
        diff = sorted(current.keys() ^ baseline.keys())
        raise SystemExit(f"the two evaluations cover different sequences: {diff[:5]}")
    out = {"smoke": [0, 0], "fp": [0, 0]}
    for sid, cur in current.items():
        base = baseline[sid]
        if cur["label"] != base["label"]:
            raise SystemExit(f"{sid}: label differs between the two evaluations")
        # Correct means alerting on smoke and staying silent on a false positive.
        want = cur["label"] == "smoke"
        cur_ok, base_ok = cur["is_positive"] == want, base["is_positive"] == want
        if cur_ok and not base_ok:
            out[cur["label"]][0] += 1
        elif base_ok and not cur_ok:
            out[cur["label"]][1] += 1
    return {label: tuple(v) for label, v in out.items()}


def test_section(cur: dict, base: dict) -> str:
    n_smoke = cur["tp"] + cur["fn"]
    n_fp = cur["fp"] + cur["tn"]
    lines = [
        "## test (pyro-dataset sequential_test, new vs main)",
        "",
        f"Sequences: {cur['num_sequences']} ({n_smoke} smoke, {n_fp} fp). "
        "Both models are scored in this run, on the same sequences. "
        "Intervals are Wilson 95%.",
        "",
        "| Metric | main | current | Δ |",
        "|--------|------|---------|---|",
    ]
    for label, key, is_ratio, ci in ROWS:
        old, new = get(base, key), get(cur, key)
        old_cell = fmt(old, is_ratio, ci and base.get(ci))
        new_cell = fmt(new, is_ratio, ci and cur.get(ci))
        lines.append(
            f"| {label} | {old_cell} | {new_cell} | {delta(new, old, is_ratio)} |"
        )
    return "\n".join(lines)


def paired_section(pairs: dict[str, tuple]) -> str:
    lines = [
        "## Paired comparison (same sequences)",
        "",
        "| | fixed by current | broken by current | McNemar exact p |",
        "|---|---|---|---|",
    ]
    names = {"smoke": "smoke (caught / missed)", "fp": "fp (silenced / alerted)"}
    for label, name in names.items():
        fixed, broken = pairs[label]
        p = mcnemar_exact_p(fixed, broken)
        lines.append(f"| {name} | {fixed} | {broken} | {p:.3f} |")
    lines += [
        "",
        "Only the sequences the two models decide differently count. A large p means "
        "the difference is within noise, not that the models are equivalent.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dataset-rev", default="unchanged")
    parser.add_argument("--result-branch", default="")
    parser.add_argument(
        "--baseline-ref", default="", help="Commit of main the baseline ran on."
    )
    args = parser.parse_args()

    for d in (args.current_dir, args.baseline_dir):
        if not (d / "metrics.json").is_file() or not (d / "predictions.json").is_file():
            print(f"missing metrics.json or predictions.json in {d}", file=sys.stderr)
            return 1
    cur_metrics, cur_preds = load_eval(args.current_dir)
    base_metrics, base_preds = load_eval(args.baseline_dir)

    footer = (
        f"**Branch:** `{args.result_branch}` | **Dataset rev:** `{args.dataset_rev}`"
    )
    if args.baseline_ref:
        footer += f" | **Baseline:** main @ `{args.baseline_ref}`"
    sections = [
        test_section(cur_metrics, base_metrics),
        paired_section(paired(cur_preds, base_preds)),
        footer,
    ]
    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
