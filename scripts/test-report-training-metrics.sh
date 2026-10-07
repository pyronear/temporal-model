#!/usr/bin/env bash
# Contract test for report-training-metrics.py.
#
# Feeds two fixture evaluations of the same test set (metrics.json +
# predictions.json each) and asserts the Markdown body has the expected rows,
# deltas, paired counts and McNemar p-values, and fails on unpaired inputs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/report-training-metrics.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/cur" "$WORK/base"

cat > "$WORK/cur/metrics.json" <<'JSON'
{"num_sequences": 4, "tp": 2, "fp": 0, "fn": 0, "tn": 2,
 "recall": 1.0, "fpr": 0.0, "recall_ci95": [0.342, 1.0], "fpr_ci95": [0.0, 0.658],
 "detected_within_frames": {"2": 0.5, "3": 1.0, "5": 1.0},
 "fpr_at_recall_95": 0.0, "roc_auc": 1.0, "median_ttd_frames": 1}
JSON
cat > "$WORK/base/metrics.json" <<'JSON'
{"num_sequences": 4, "tp": 1, "fp": 1, "fn": 1, "tn": 1,
 "recall": 0.5, "fpr": 0.5, "recall_ci95": [0.095, 0.905], "fpr_ci95": [0.095, 0.905],
 "detected_within_frames": {"2": 0.5, "3": 0.5, "5": 0.5},
 "fpr_at_recall_95": 0.5, "roc_auc": 0.75, "median_ttd_frames": 0}
JSON
# current fixes s2 (missed -> caught) and f1 (alert -> silent).
cat > "$WORK/cur/predictions.json" <<'JSON'
[{"sequence_id": "s1", "label": "smoke", "is_positive": true},
 {"sequence_id": "s2", "label": "smoke", "is_positive": true},
 {"sequence_id": "f1", "label": "fp", "is_positive": false},
 {"sequence_id": "f2", "label": "fp", "is_positive": false}]
JSON
cat > "$WORK/base/predictions.json" <<'JSON'
[{"sequence_id": "f2", "label": "fp", "is_positive": false},
 {"sequence_id": "s1", "label": "smoke", "is_positive": true},
 {"sequence_id": "s2", "label": "smoke", "is_positive": false},
 {"sequence_id": "f1", "label": "fp", "is_positive": true}]
JSON

run() { python3 "$SCRIPT" "$@"; }

BODY=$(run --current-dir "$WORK/cur" --baseline-dir "$WORK/base" \
           --dataset-rev v4.5.0 --result-branch result_v4.5.0)
expect() {
  echo "$BODY" | grep -qF "$1" || { echo "FAIL: $2"; echo "$BODY"; exit 1; }
}
expect "| Recall | 0.5000 [0.095, 0.905] | 1.0000 [0.342, 1.000] | +0.5000 |" "recall row with CIs"
expect "| False alerts (FP) | 1 | 0 | -1 |" "FP count row"
expect "| Detected within 3 frames | 0.5000 | 1.0000 | +0.5000 |" "detection-within row"
expect "| smoke (caught / missed) | 1 | 0 | 1.000 |" "paired smoke row"
expect "| fp (silenced / alerted) | 1 | 0 | 1.000 |" "paired fp row"
expect "**Dataset rev:** \`v4.5.0\`" "dataset rev footer"
if echo "$BODY" | grep -q "Precision"; then
  echo "FAIL: precision is prevalence-dependent and must not be reported"; exit 1
fi

# McNemar exact p on known counts: 10 vs 0 discordant -> 2 * 0.5^10.
P=$(python3 -c "
import importlib.util
s = importlib.util.spec_from_file_location('r', '$SCRIPT'); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(round(m.mcnemar_exact_p(10, 0), 6), m.mcnemar_exact_p(0, 0), m.mcnemar_exact_p(3, 3))")
[ "$P" = "0.001953 1.0 1.0" ] || { echo "FAIL: mcnemar p-values: $P"; exit 1; }

# --- unpaired inputs fail loudly ---
echo '[{"sequence_id": "s1", "label": "smoke", "is_positive": true}]' > "$WORK/base/predictions.json"
if run --current-dir "$WORK/cur" --baseline-dir "$WORK/base" >/dev/null 2>&1; then
  echo "FAIL: different sequence sets should exit non-zero"; exit 1
fi

# --- missing files fail loudly ---
rm "$WORK/base/predictions.json"
if run --current-dir "$WORK/cur" --baseline-dir "$WORK/base" >/dev/null 2>&1; then
  echo "FAIL: missing predictions.json should exit non-zero"; exit 1
fi

echo "OK"
