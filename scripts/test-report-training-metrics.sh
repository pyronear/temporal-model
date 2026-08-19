#!/usr/bin/env bash
# Contract test for report-training-metrics.py.
#
# Feeds fixture metrics snapshots (with and without a baseline) and asserts
# the Markdown body contains the expected rows, deltas, and fallbacks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/report-training-metrics.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/results" "$WORK/baseline" "$WORK/empty"

cat > "$WORK/results/pyro-annotator.json" <<'JSON'
{"num_sequences": 317, "tp": 42, "fp": 86, "fn": 2, "tn": 187,
 "precision": 0.3281, "recall": 0.9545, "f1": 0.4884, "fpr": 0.315,
 "mean_ttd_frames": 3.4, "median_ttd_frames": 3,
 "pr_auc": 0.6902, "roc_auc": 0.8967}
JSON
cat > "$WORK/baseline/pyro-annotator.json" <<'JSON'
{"num_sequences": 317, "tp": 43, "fp": 114, "fn": 1, "tn": 159,
 "precision": 0.2739, "recall": 0.9773, "f1": 0.4279, "fpr": 0.4176,
 "mean_ttd_frames": 2.6, "median_ttd_frames": 1,
 "pr_auc": 0.7294, "roc_auc": 0.9011}
JSON
for split in train val; do
  cat > "$WORK/results/$split.json" <<'JSON'
{"num_sequences": 350, "tp": 167, "fp": 11, "fn": 8, "tn": 164,
 "precision": 0.9382, "recall": 0.9543, "f1": 0.9462, "fpr": 0.0629,
 "roc_auc": 0.9751}
JSON
done

run() { python3 "$SCRIPT" "$@"; }

# --- with baseline: delta column is computed ---
BODY=$(run --results-dir "$WORK/results" --baseline-dir "$WORK/baseline" \
           --dataset-rev v4.1.0 --result-branch result_v4.1.0)
echo "$BODY" | grep -qF "| False alerts (FP) | 114 | 86 | -28 |" \
  || { echo "FAIL: FP delta row"; echo "$BODY"; exit 1; }
echo "$BODY" | grep -qF "| Precision | 0.2739 | 0.3281 | +0.0542 |" \
  || { echo "FAIL: precision delta row"; echo "$BODY"; exit 1; }
echo "$BODY" | grep -qF "**Dataset rev:** \`v4.1.0\`" \
  || { echo "FAIL: dataset rev footer"; echo "$BODY"; exit 1; }
echo "$BODY" | grep -q "train (in-sample)" \
  || { echo "FAIL: train split row"; echo "$BODY"; exit 1; }

# --- without baseline: n/a deltas + explicit note ---
BODY=$(run --results-dir "$WORK/results" --baseline-dir "$WORK/empty" \
           --dataset-rev unchanged --result-branch result_x)
echo "$BODY" | grep -qF "(no baseline on main yet)" \
  || { echo "FAIL: no-baseline note"; echo "$BODY"; exit 1; }
echo "$BODY" | grep -qF "| False alerts (FP) | n/a | 86 | n/a |" \
  || { echo "FAIL: no-baseline FP row"; echo "$BODY"; exit 1; }

# --- missing snapshot: fails loudly ---
rm "$WORK/results/val.json"
if run --results-dir "$WORK/results" --baseline-dir "$WORK/baseline" >/dev/null 2>&1; then
  echo "FAIL: missing snapshot should exit non-zero"; exit 1
fi

echo "OK"
