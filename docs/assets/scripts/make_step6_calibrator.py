"""Step 6 visuals: how the logistic calibrator maps tube features to probability."""
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/assets"

with zipfile.ZipFile(ROOT / "api/models/model.zip") as zf:
    cal = json.loads(zf.read("logistic_calibrator.json"))
    import yaml
    cfg = yaml.safe_load(zf.read("config.yaml"))

W = np.asarray(cal["coefficients"])  # [logit, log_len, mean_conf, n_tubes]
B = float(cal["intercept"])
THR = float(cfg["decision"]["logistic_threshold"])

def proba(logit, length, mean_conf, n_tubes):
    z = W @ np.array([logit, math.log1p(length), mean_conf, float(n_tubes)]) + B
    return 1.0 / (1.0 + np.exp(-z))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)

# ── panel A: probability vs raw logit for three tube contexts ────────────────
logits = np.linspace(-10, 12, 300)
contexts = [
    ("2 frames, mean conf 0.15, 5 tubes in scene", 2, 0.15, 5, "tab:red"),
    ("8 frames, mean conf 0.30, 2 tubes in scene", 8, 0.30, 2, "tab:orange"),
    ("20 frames, mean conf 0.55, 1 tube in scene", 20, 0.55, 1, "tab:green"),
]
for label, length, mc, nt, color in contexts:
    ps = [proba(x, length, mc, nt) for x in logits]
    ax1.plot(logits, ps, color=color, lw=2, label=label)
ax1.axhline(THR, color="0.3", ls="--", lw=1)
ax1.text(-9.7, THR + 0.03, "logistic_threshold", fontsize=8, color="0.3")
ax1.set_xlabel("raw classifier logit")
ax1.set_ylabel("calibrated probability")
ax1.set_title("same logit, different evidence", fontsize=10)
ax1.legend(fontsize=7.5, loc="lower right")
ax1.set_ylim(-0.02, 1.02)

# ── panel B: decision boundary over logit × tube length ─────────────────────
lo = np.linspace(-10, 12, 220)
ln = np.arange(2, 21)
P = np.array([[proba(x, length, 0.30, 2) for x in lo] for length in ln])
im = ax2.imshow(P, aspect="auto", origin="lower", cmap="RdYlGn",
                extent=[lo[0], lo[-1], ln[0], ln[-1]], vmin=0, vmax=1)
cs = ax2.contour(lo, ln, P, levels=[THR], colors="black", linewidths=1.5,
                 linestyles="--")
ax2.clabel(cs, fmt={THR: "decision boundary"}, fontsize=8)
ax2.set_xlabel("raw classifier logit")
ax2.set_ylabel("tube length (frames)")
ax2.set_title("longer tubes need less logit (conf 0.30, 2 tubes)", fontsize=10)
fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label="probability")

fig.tight_layout()
fig.savefig(OUT / "calibrator_curves.png", bbox_inches="tight")
print("wrote calibrator figure; threshold", THR)
