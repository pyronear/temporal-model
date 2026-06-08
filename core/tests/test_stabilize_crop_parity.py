"""End-to-end stabilize crop parity: training crop == inference crop on a gappy tube.

The training path (``model_input.process_tube`` → PNG) and the inference path
(``inference.crop_tube_patches`` → tensor) both crop through the shared
``stabilize.tube_window`` helper, but each feeds it through its OWN adapter layer
(dict entries with ``e["bbox"]``/``e["is_gap"]`` vs ``TubeEntry``/``Detection``).
The window *policy* is shared, so it cannot drift between the two paths; the
adapters are NOT shared and can. This test guards **adapter parity**: it feeds both
paths the same tube — including a gap frame whose box sits far outside the observed
boxes — and asserts identical patches. If one adapter projected entries differently
from the other (e.g. dropped ``is_gap`` so it stopped excluding the far gap box, or
mismapped the bbox fields), the two windows would diverge and the patches would
differ; the far gap box makes such a divergence visible rather than negligible.

Scope: this does NOT verify ``tube_window``'s exclusion policy itself — a bug there
would shift BOTH paths identically and still pass here. That policy (non-gap union,
fallback, None handling) is unit-tested in ``test_stabilize.py``.
"""

import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from temporal_model.core.inference import crop_tube_patches
from temporal_model.core.model_input import process_tube
from temporal_model.core.protocol import Frame
from temporal_model.core.types import Detection, Tube, TubeEntry

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Frame 1 is a gap with a far-away box (cx=0.9). Observed boxes (cx 0.4 / 0.6) give
# a centered window; an adapter that mishandled the gap (included its box) would
# shift the window hard right, so any adapter divergence between the two paths is
# visible rather than negligible.
_BOXES = [
    ((0.4, 0.5, 0.1, 0.1), False),
    ((0.9, 0.5, 0.2, 0.2), True),
    ((0.6, 0.5, 0.1, 0.1), False),
]


def test_stabilize_crop_parity_offline_vs_inference_gappy_tube(tmp_path):
    # Three identical horizontal-gradient frames: a different crop window yields a
    # visibly different patch, so identical patches prove identical windows.
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    img = np.stack([ramp, ramp, ramp], axis=-1)
    seq_id = "site_1_2023-09-01T00-00-00"
    images_dir = tmp_path / "raw" / "wildfire" / seq_id / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [f"{seq_id}_f{i}" for i in range(3)]
    for fid in frame_ids:
        Image.fromarray(img).save(images_dir / f"{fid}.jpg", format="JPEG", quality=95)

    # Offline path: write a tube record and run process_tube -> PNG patches.
    record = {
        "sequence_id": seq_id,
        "split": "train",
        "label": "smoke",
        "source": "gt",
        "num_frames": 3,
        "tube": {
            "start_frame": 0,
            "end_frame": 2,
            "entries": [
                {
                    "frame_idx": i,
                    "frame_id": frame_ids[i],
                    "bbox": list(box),
                    "is_gap": is_gap,
                    "confidence": 0.0 if is_gap else 0.9,
                }
                for i, (box, is_gap) in enumerate(_BOXES)
            ],
        },
    }
    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    tube_path.write_text(json.dumps(record))
    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
        stabilize=True,
    )

    # Inference path: same tube as TubeEntry/Detection objects.
    tube = Tube(
        tube_id=0,
        entries=[
            TubeEntry(
                frame_idx=i,
                detection=Detection(
                    class_id=0,
                    cx=box[0],
                    cy=box[1],
                    w=box[2],
                    h=box[3],
                    confidence=0.0 if is_gap else 0.9,
                ),
                is_gap=is_gap,
            )
            for i, (box, is_gap) in enumerate(_BOXES)
        ],
        start_frame=0,
        end_frame=2,
    )
    frames = [
        Frame(frame_id=fid, image_path=images_dir / f"{fid}.jpg", timestamp=None)
        for fid in frame_ids
    ]
    patches, mask = crop_tube_patches(
        tube,
        frames,
        context_factor=1.5,
        patch_size=224,
        max_frames=5,
        normalization_mean=_MEAN,
        normalization_std=_STD,
        stabilize=True,
    )

    assert mask[:3].all()
    mean_t = torch.tensor(_MEAN).view(3, 1, 1)
    std_t = torch.tensor(_STD).view(3, 1, 1)
    for slot in range(3):
        png = out_dir / seq_id / f"frame_{slot:02d}.png"
        offline = (to_tensor(Image.open(png).convert("RGB")) - mean_t) / std_t
        assert torch.allclose(patches[slot], offline, atol=1e-6), f"slot {slot} differs"
