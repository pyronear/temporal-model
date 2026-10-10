"""Raspberry Pi 5 feasibility check for classifier (step 5) architecture variants.

Latency and peak RAM do not depend on trained weights, so every variant is
built with random weights, exported to ONNX, and timed with onnxruntime — the
torch-free runtime an edge device would use. Two subcommands:

    # 1. on a dev machine (needs the torch stack: `cd benchmark && uv sync`)
    uv run python scripts/pi5_variants.py export --out pi5_models

    # 2. on the Pi (`pip install numpy onnxruntime pillow`)
    python pi5_variants.py bench pi5_models --images <sequence_dir>

Variants (all max_frames=20, fused into the existing transformer head by
adding a projected embedding to each frame token):

- ``vit224``            current production classifier (DINOv2 ViT-S/14 on 224 crops)
- ``vit224+yolo_p5``    + ROI-pooled YOLO P5 embedding per tube frame
- ``vit224+yolo_p345``  + ROI-pooled YOLO P3+P4+P5 embeddings (concatenated)
- ``vit224+ctx_vit``    + full frame through the same ViT (scene context)
- ``vit224+ctx_yolo``   + global-pooled YOLO P5 of the full frame (free context)
- ``yolo_p345_only``    no ViT: YOLO ROI embeddings + head only
- ``vit112``            current classifier on 112 crops (4x fewer ViT tokens)

Plus ``yolo11s_1024``: the companion detector at production resolution, also
returning its three neck feature maps (one frame per call; the API caches
detections, so in steady state only new frames pay this).

YOLO embeddings enter the classifier already pooled ([N, T, C]): the cheap way
to ship them is to ROI-pool each detection box right after the YOLO forward
and cache that vector next to the box, instead of caching the feature maps.
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

MAX_FRAMES = 20
YOLO_IMGSZ = 1024


# --------------------------------------------------------------------- export


def _export_yolo(out: Path, weights: Path | None) -> list[int]:
    """Export YOLO11s with its neck maps as extra outputs; return their channels."""
    import torch  # noqa: PLC0415  # export-only dependency
    from ultralytics import YOLO  # noqa: PLC0415

    if weights is None:
        from temporal_model.core.fetch_detector import fetch_detector  # noqa: PLC0415

        weights = fetch_detector(out / "yolo_weights.pt")
    det = YOLO(str(weights)).model.float().eval()
    det.model[-1].export = True  # Detect returns the raw prediction tensor

    class YoloTaps(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = []
            for layer in self.m.model:
                if layer.f != -1:
                    x = (
                        y[layer.f]
                        if isinstance(layer.f, int)
                        else [x if j == -1 else y[j] for j in layer.f]
                    )
                if layer is self.m.model[-1]:
                    taps = list(x)  # P3, P4, P5 feeding the Detect head
                x = layer(x)
                y.append(x if layer.i in self.m.save else None)
            return (x, *taps)

    wrapper = YoloTaps(det)
    dummy = torch.zeros(1, 3, YOLO_IMGSZ, YOLO_IMGSZ)
    with torch.no_grad():
        taps = wrapper(dummy)[1:]
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out / "yolo11s_1024.onnx"),
        input_names=["image"],
        output_names=["pred", "p3", "p4", "p5"],
        opset_version=18,
        dynamo=False,
    )
    return [int(t.shape[1]) for t in taps]


DINOV2_S14 = "vit_small_patch14_dinov2.lvd142m"
DINOV3_S16 = "vit_small_patch16_dinov3.lvd1689m"


def _build_variant(
    img: int, yolo_dim: int, ctx: str | None, use_vit: bool, backbone: str = DINOV2_S14
):
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    from temporal_model.core.temporal_classifier import (  # noqa: PLC0415
        TemporalSmokeClassifier,
    )

    clf = TemporalSmokeClassifier(
        backbone=backbone,
        pretrained=False,
        # ConvNeXt has no CLS token and a fixed-size-agnostic stem.
        global_pool="token" if backbone.startswith("vit_") else "avg",
        img_size=img if backbone.startswith("vit_") else None,
        max_frames=MAX_FRAMES,
    )
    d = clf.backbone.feat_dim

    class Variant(nn.Module):
        def __init__(self, ctx_dim: int):
            super().__init__()
            self.clf = clf
            self.yolo_proj = nn.Linear(yolo_dim, d) if yolo_dim else None
            self.ctx_proj = nn.Linear(ctx_dim, d) if ctx else None

        def forward(self, mask, *extra):
            extra = list(extra)
            n = mask.shape[0]
            feats = 0
            if use_vit:
                patches = extra.pop(0)
                feats = self.clf.backbone(patches.flatten(0, 1))
                feats = feats.reshape(n, MAX_FRAMES, d)
            if self.yolo_proj is not None:
                feats = feats + self.yolo_proj(extra.pop(0))
            if ctx == "vit":
                feats = feats + self.ctx_proj(self.clf.backbone(extra.pop(0)))[None]
            elif ctx == "yolo":
                feats = feats + self.ctx_proj(extra.pop(0))[None]
            return self.clf.head(feats, mask)

    model = Variant(ctx_dim=d if ctx == "vit" else yolo_dim).eval()
    for module in model.modules():  # see core export_onnx.export_classifier
        if hasattr(module, "fused_attn"):
            module.fused_attn = False

    # Symbolic "N" = number of tubes (batch axis); context inputs are per frame.
    specs = {"mask": (["N", MAX_FRAMES], "bool")}
    if use_vit:
        specs["patches"] = (["N", MAX_FRAMES, 3, img, img], "float32")
    if yolo_dim:
        specs["yolo_emb"] = (["N", MAX_FRAMES, yolo_dim], "float32")
    if ctx == "vit":
        specs["ctx_frames"] = ([MAX_FRAMES, 3, img, img], "float32")
    elif ctx == "yolo":
        specs["ctx_emb"] = ([MAX_FRAMES, yolo_dim], "float32")
    return model, specs, torch


def _export_variant(out: Path, name: str, **kw) -> dict:
    model, specs, torch = _build_variant(**kw)
    dummy = []
    for shape, dtype in specs.values():
        concrete = [2 if s == "N" else s for s in shape]  # batch-2 avoids folding N
        dummy.append(
            torch.ones(concrete, dtype=torch.bool)
            if dtype == "bool"
            else torch.randn(concrete)
        )
    path = out / f"{name}.onnx"
    torch.onnx.export(
        model,
        tuple(dummy),
        str(path),
        input_names=list(specs),
        output_names=["logit"],
        dynamic_axes={k: {0: "N"} for k, (s, _) in specs.items() if s[0] == "N"},
        opset_version=18,
        dynamo=False,
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    quantize_dynamic(
        path,
        out / f"{name}.int8.onnx",
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],  # ConvInteger is slow on ORT CPU
    )
    return {k: {"shape": s, "dtype": t} for k, (s, t) in specs.items()}


def cmd_export(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if args.only else {}
    if "yolo11s_1024" in manifest:  # --only re-run: keep the exported detector
        p3, p4, p5 = manifest["yolo11s_1024"]["neck_channels"]
    else:
        p3, p4, p5 = _export_yolo(out, args.yolo_weights)
        manifest["yolo11s_1024"] = {
            "inputs": {
                "image": {"shape": [1, 3, YOLO_IMGSZ, YOLO_IMGSZ], "dtype": "float32"}
            },
            "neck_channels": [p3, p4, p5],
        }
    p345 = p3 + p4 + p5
    v3 = dict(yolo_dim=0, ctx=None, use_vit=True)
    variants = {
        "vit224": dict(img=224, yolo_dim=0, ctx=None, use_vit=True),
        "vit224+yolo_p5": dict(img=224, yolo_dim=p5, ctx=None, use_vit=True),
        "vit224+yolo_p345": dict(img=224, yolo_dim=p345, ctx=None, use_vit=True),
        "vit224+ctx_vit": dict(img=224, yolo_dim=0, ctx="vit", use_vit=True),
        "vit224+ctx_yolo": dict(img=224, yolo_dim=p5, ctx="yolo", use_vit=True),
        "yolo_p345_only": dict(img=224, yolo_dim=p345, ctx=None, use_vit=False),
        "vit112": dict(img=112, yolo_dim=0, ctx=None, use_vit=True),
        # DINOv3 backbones (patch 16: 196 tokens at 224 vs 256 for DINOv2/14)
        "dinov3_s16_224": dict(img=224, backbone=DINOV3_S16, **v3),
        "dinov3_s16_112": dict(img=112, backbone=DINOV3_S16, **v3),
        "dinov3_splus16_224": dict(
            img=224, backbone="vit_small_plus_patch16_dinov3.lvd1689m", **v3
        ),
        "dinov3_convnext_t_224": dict(
            img=224, backbone="convnext_tiny.dinov3_lvd1689m", **v3
        ),
    }
    for name, kw in variants.items():
        if args.only and name not in args.only:
            continue
        print(f"exporting {name}", flush=True)
        manifest[name] = {"inputs": _export_variant(out, name, **kw)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}  (copy this directory and this script to the Pi)")


# ---------------------------------------------------------------------- bench

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _real_frames(seq_dir: Path, img: int) -> dict:
    """Real inputs from a pyro-dataset sequence dir (``images/`` + YOLO ``labels/``).

    Crops are one fixed square window (label box x ``context`` 1.5) over the
    first ``MAX_FRAMES`` frames, like the stabilized production crop; ponytail:
    first labelled box stands in for the tube window, enough for timing.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    paths = sorted((seq_dir / "images").glob("*.jpg"))[:MAX_FRAMES]
    paths += paths[-1:] * (MAX_FRAMES - len(paths))  # pad short sequences
    box = (0.5, 0.5, 0.05, 0.05)
    for lbl in sorted((seq_dir / "labels").glob("*.txt")):
        line = lbl.read_text().split("\n")[0].split()
        if len(line) == 5:
            box = tuple(map(float, line[1:]))
            break
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]

    def norm(im):
        a = np.asarray(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
        return (a / 255 - mean) / std

    crops, full = [], []
    for path in paths:
        im = Image.open(path)
        w, h = im.size
        cx, cy, bw, bh = box
        side = max(bw * w, bh * h, 32) * 1.5
        x0, y0 = cx * w - side / 2, cy * h - side / 2
        crop = im.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))
        crops.append(norm(crop.resize((img, img), Image.BILINEAR)))
        full.append(norm(im.resize((img, img), Image.BILINEAR)))
    # YOLO input: letterbox the first frame to YOLO_IMGSZ, 0..1 scaling.
    im = Image.open(paths[0]).convert("RGB")
    scale = YOLO_IMGSZ / max(im.size)
    im = im.resize((round(im.width * scale), round(im.height * scale)))
    canvas = Image.new("RGB", (YOLO_IMGSZ, YOLO_IMGSZ), (114, 114, 114))
    canvas.paste(im, ((YOLO_IMGSZ - im.width) // 2, (YOLO_IMGSZ - im.height) // 2))
    yolo = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    return {
        "patches": np.stack(crops)[None],
        "ctx_frames": np.stack(full),
        "image": yolo,
    }


def _run_one(
    path: Path, inputs_spec: dict, n: int, threads: int, runs: int, images: str
) -> dict:
    """Time one model in this (fresh) process; ru_maxrss is then its own peak."""
    import numpy as np  # noqa: PLC0415  # keep `export` usable without these
    import onnxruntime as ort  # noqa: PLC0415

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    real = {}
    if images:
        img = inputs_spec.get("patches", inputs_spec.get("ctx_frames"))
        real = _real_frames(Path(images), img["shape"][-1] if img else 224)
    feed = {}
    for name, spec in inputs_spec.items():
        shape = [n if s == "N" else s for s in spec["shape"]]
        if spec["dtype"] == "bool":
            feed[name] = np.ones(shape, dtype=bool)
        elif name in real:  # YOLO embeddings stay random: content-free for timing
            feed[name] = np.ascontiguousarray(np.broadcast_to(real[name], shape))
        else:
            feed[name] = rng.standard_normal(shape, dtype=np.float32)
    sess.run(None, feed)  # warmup
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024 / (1024 if sys.platform == "darwin" else 1)  # macOS: bytes
    return {
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(min(times), 1),
        "peak_rss_mb": round(rss_mb),
        "load_s": round(load_s, 2),
        "file_mb": round(path.stat().st_size / 1e6, 1),
    }


def cmd_bench(args: argparse.Namespace) -> None:
    manifest = json.loads((args.models / "manifest.json").read_text())
    import onnxruntime as ort  # noqa: PLC0415

    machine = {
        "host": platform.node(),
        "machine": platform.machine(),
        "cpus": __import__("os").cpu_count(),
        "onnxruntime": ort.__version__,
        "threads": args.threads or "ort default",
    }
    print(json.dumps(machine))
    rows = []
    if args.images:
        times = []
        for _ in range(3):
            t = time.perf_counter()
            _real_frames(args.images, 224)
            times.append((time.perf_counter() - t) * 1000)
        rows.append(
            {
                "model": "preprocess_20_frames",
                "median_ms": round(statistics.median(times), 1),
            }
        )
        print(
            f"preprocess (decode+crop+resize {MAX_FRAMES} frames): "
            f"{rows[-1]['median_ms']} ms",
            flush=True,
        )
    for name, entry in manifest.items():
        if args.only and name not in args.only:
            continue
        per_tube = any("N" in s["shape"] for s in entry["inputs"].values())
        for suffix in ("", ".int8"):
            path = args.models / f"{name}{suffix}.onnx"
            if not path.exists():
                continue
            for n in args.tubes if per_tube else [1]:
                cmd = [
                    sys.executable, __file__, "_run-one", str(path),
                    json.dumps(entry["inputs"]), str(n), str(args.threads),
                    str(args.runs), str(args.images or ""),
                ]  # fmt: skip
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode:
                    print(f"{name}{suffix} N={n} FAILED\n{res.stderr[-2000:]}")
                    continue
                row = {"model": f"{name}{suffix}", "tubes": n if per_tube else "-"}
                row |= json.loads(res.stdout)
                rows.append(row)
                print(
                    f"{row['model']:<26} tubes={row['tubes']!s:<2} "
                    f"median={row['median_ms']:>8} ms  min={row['min_ms']:>8} ms  "
                    f"peak_rss={row['peak_rss_mb']:>5} MB  file={row['file_mb']} MB",
                    flush=True,
                )
    if "yolo11s_1024" in manifest:
        p3, p4, p5 = manifest["yolo11s_1024"]["neck_channels"]
        maps_mb = (
            4
            * sum(c * (YOLO_IMGSZ // s) ** 2 for c, s in ((p3, 8), (p4, 16), (p5, 32)))
            / 1e6
        )
        print(
            f"note: caching raw P3+P4+P5 maps costs {maps_mb:.1f} MB/frame "
            f"({maps_mb * MAX_FRAMES:.0f} MB per {MAX_FRAMES}-frame camera window) "
            f"vs {4 * (p3 + p4 + p5) / 1e3:.1f} KB per ROI-pooled detection"
        )
    out = args.models / f"results-{platform.node()}.json"
    out.write_text(json.dumps({"machine": machine, "rows": rows}, indent=2))
    print(f"results -> {out}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_run-one":  # internal subprocess entry
        path, spec, n, threads, runs, images = sys.argv[2:8]
        res = _run_one(
            Path(path), json.loads(spec), int(n), int(threads), int(runs), images
        )
        print(json.dumps(res))
        return

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="build + export every variant (needs torch)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="+", help="re-export a subset into --out")
    p.add_argument(
        "--yolo-weights",
        type=Path,
        help="detector .pt (default: fetch the pinned one from HuggingFace)",
    )
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("bench", help="time every exported model (needs onnxruntime)")
    p.add_argument("models", type=Path, help="directory written by `export`")
    p.add_argument("--tubes", type=int, nargs="+", default=[1, 3])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    p.add_argument("--only", nargs="+", help="subset of model names")
    p.add_argument(
        "--images",
        type=Path,
        help="pyro-dataset sequence dir (images/ + labels/) for real inputs",
    )
    p.set_defaults(func=cmd_bench)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
