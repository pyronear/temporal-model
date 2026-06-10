"""Render the actual ViT-backbone and transformer-head inputs for one tube.

Uses the released model.zip (v0.1.0) and the pre-cropped stabilized patches of
the brison sequence (train/data/05_model_input).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from temporal_model.core.package import load_model_package

ROOT = Path(__file__).resolve().parents[3]
SEQ = "pyronear-sdis-07_brison_290_2024-02-03T11-13-08"
MI = ROOT / "train/data/05_model_input/train" / SEQ
OUT = ROOT / "docs/assets"

pkg = load_model_package(
    ROOT / "api/models/model.zip", extract_dir=ROOT / ".cache/temporal_model_core"
)
mi = pkg.model_input
mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)

meta = json.loads((MI / "meta.json").read_text())
T = meta["num_frames"]
patches = torch.zeros(T, 3, 224, 224)
for f in meta["frames"]:
    img = Image.open(MI / f["filename"]).convert("RGB")
    patches[f["frame_idx"]] = (to_tensor(img) - mean) / std
mask = torch.ones(T, dtype=torch.bool)

clf = pkg.classifier.eval()
with torch.no_grad():
    flat = patches.reshape(T * 1, 3, 224, 224)
    feats = clf.backbone(flat).reshape(T, -1)  # [20, 384]
    logit = clf(patches.unsqueeze(0), mask.unsqueeze(0)).item()
print(f"tube logit: {logit:.3f}")


# ── figure A: normalized patches (the backbone's actual input) ──────────────
def font(size):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
    )


idxs = [0, 6, 12, 19]
tiles = []
for i in idxs:
    x = patches[i].clamp(-2.5, 2.5)
    x = ((x + 2.5) / 5.0 * 255).byte().permute(1, 2, 0).numpy()
    p = Image.fromarray(x)
    d = ImageDraw.Draw(p)
    d.rectangle([0, 0, 64, 24], fill=(0, 0, 0))
    d.text((6, 3), f"t = {i}", fill="white", font=font(16))
    tiles.append(p)
gap = 4
strip = Image.new("RGB", (224 * len(tiles) + gap * (len(tiles) - 1), 224), "white")
x0 = 0
for t in tiles:
    strip.paste(t, (x0, 0))
    x0 += 224 + gap
strip.save(OUT / "brison_backbone_input.jpg", quality=92)

# ── figure B: per-frame embeddings (the head's actual input) ────────────────
fig, ax = plt.subplots(figsize=(11, 3.2), dpi=150)
im = ax.imshow(feats.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xlabel("feature dimension (384, from the ViT backbone)")
ax.set_ylabel("time step t")
ax.set_yticks([0, 5, 10, 15, 19])
ax.set_title(
    "one tube as the transformer head receives it — "
    f"20 embeddings × 384 dims (tube logit: {logit:+.2f})"
)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
fig.tight_layout()
fig.savefig(OUT / "brison_head_input.png")
print("wrote figures")
