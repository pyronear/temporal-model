"""Timeline v2: rows grouped by fate, merge groups banded + connected."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.tubes import build_tubes, merge_colocated_tubes
from temporal_model.core.inference import filter_and_interpolate_tubes

ROOT = Path(__file__).resolve().parents[3]
SEQ = "pyronear-sdis-07_brison_020_2024-01-18T14-18-41"
RAW = ROOT / "train/data/01_raw/datasets/train/wildfire" / SEQ / "images"
OUT = ROOT / "docs/assets"

model = BboxTubeTemporalModel.from_package(ROOT / "api/models/model.zip")
tc = model._cfg["tubes"]
paths = sorted(RAW.glob("*.jpg"))[: model._cfg["classifier"]["max_frames"]]
fd = model.detect(model.load_sequence(paths))

candidates = build_tubes(fd, iou_threshold=tc["iou_threshold"], max_misses=tc["max_misses"])
f1 = filter_and_interpolate_tubes(
    candidates, min_tube_length=tc["infer_min_tube_length"],
    min_detected_entries=tc["min_detected_entries"], interpolate_gaps=False)
kept = filter_and_interpolate_tubes(
    merge_colocated_tubes(f1, merge_iomin=tc["merge_iomin"],
                          merge_prox_factor=tc["merge_prox_factor"],
                          merge_max_gap=tc["merge_max_gap"]),
    min_tube_length=tc["infer_min_tube_length"],
    min_detected_entries=tc["min_detected_entries"], interpolate_gaps=True)

def obs_keys(t):
    return {(e.frame_idx, round(e.detection.cx, 9), round(e.detection.cy, 9))
            for e in t.entries if e.detection is not None and not e.is_gap}

f1_ids = {t.tube_id for t in f1}
kept_keys = [(k.tube_id, obs_keys(k)) for k in kept]

def fate(cand):
    if cand.tube_id in f1_ids:
        ck = obs_keys(cand)
        for kid, kk in kept_keys:
            if ck & kk:
                return kid
    return None

fates = [fate(c) for c in candidates]
groups: dict = {}
for i, kid in enumerate(fates):
    groups.setdefault(kid, []).append(i)

row_order = []
for t in kept:
    row_order += sorted(groups.get(t.tube_id, []), key=lambda i: candidates[i].start_frame)
row_order += groups.get(None, [])
row_of = {ci: r for r, ci in enumerate(row_order)}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 5.8), dpi=150, sharex=True,
    gridspec_kw={"height_ratios": [len(candidates), len(kept) + 1]})

GRAY = "0.72"
for ci in row_order:
    cand, row, kid = candidates[ci], row_of[ci], fates[ci]
    color = GRAY if kid is None else plt.cm.tab10(kid)
    xs = [e.frame_idx for e in cand.entries if e.detection is not None]
    ax1.plot(xs, [row] * len(xs), "o-", color=color, ms=5, lw=1.2, zorder=3)

for t in kept:
    idxs = groups.get(t.tube_id, [])
    if not idxs:
        continue
    color = plt.cm.tab10(t.tube_id)
    rows = [row_of[i] for i in idxs]
    x_end = max(candidates[i].end_frame for i in idxs)
    if len(idxs) > 1:
        x0 = min(candidates[i].start_frame for i in idxs)
        ax1.add_patch(FancyBboxPatch(
            (x0 - 0.35, min(rows) - 0.38), x_end - x0 + 0.7, max(rows) - min(rows) + 0.76,
            boxstyle="round,pad=0.12", facecolor=color, alpha=0.10,
            edgecolor=color, lw=1.1, ls="--", zorder=1))
        frag = sorted(idxs, key=lambda i: candidates[i].start_frame)
        for a, b in zip(frag, frag[1:]):
            ax1.plot([candidates[a].end_frame, candidates[b].start_frame],
                     [row_of[a], row_of[b]], ls=":", color=color, lw=1.6, zorder=2)
        label = f"merged → tube {t.tube_id}"
    else:
        label = f"→ tube {t.tube_id}"
    ax1.annotate(label, (x_end + 0.45, sum(rows) / len(rows)), color=color,
                 fontsize=8, fontweight="bold", va="center")

for ci in groups.get(None, []):
    cand = candidates[ci]
    ax1.annotate("✗ dropped by filter", (cand.end_frame + 0.45, row_of[ci]),
                 color=GRAY, fontsize=8, va="center")

ax1.set_yticks(range(len(row_order)))
ax1.set_yticklabels([f"cand {candidates[ci].tube_id}" for ci in row_order], fontsize=8)
ax1.invert_yaxis()
ax1.set_title(f"raw candidates from build_tubes ({len(candidates)}), "
              f"grouped by fate — dashed band = fragments fused by the merge pass",
              fontsize=10)

for row, t in enumerate(kept):
    color = plt.cm.tab10(t.tube_id)
    xs = [e.frame_idx for e in t.entries if e.detection is not None]
    ax2.plot(xs, [row] * len(xs), "-", color=color, lw=1.2, zorder=1)
    for e in t.entries:
        if e.detection is None:
            continue
        if e.is_gap:
            ax2.plot(e.frame_idx, row, "D", color="white", mec="darkorange",
                     mew=1.6, ms=6, zorder=2)
        else:
            ax2.plot(e.frame_idx, row, "o", color=color, ms=5, zorder=2)
ax2.set_yticks(range(len(kept)))
ax2.set_yticklabels([f"tube {t.tube_id}" for t in kept], fontsize=8)
ax2.invert_yaxis()
ax2.set_ylim(len(kept) - 0.5, -0.5)
ax2.set_xlabel("frame index t")
ax2.set_xticks(range(0, 20, 2))
ax2.set_xlim(-0.8, 24.5)
ax2.set_title(f"kept tubes after filter + merge + interpolate ({len(kept)}) — "
              f"orange diamonds = interpolated gap entries", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "brison020_tube_stages.png", bbox_inches="tight")
print("wrote timeline v2")
