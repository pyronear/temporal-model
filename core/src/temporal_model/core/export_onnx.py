"""Export the torch tube classifier to ONNX and bundle it as ``model_onnx.zip``.

The ONNX package is the torch-free runtime artifact consumed by
:class:`~temporal_model.core.onnx_model.OnnxTemporalModel`. It is derived from
a released ``model.zip`` and published next to it under the same HuggingFace
tag. Contents (format 2):

- ``manifest.yaml`` — format version, file pointers, source ``model.zip``
  identity (SHA-256) and the ONNX I/O contract (names, shapes, dtypes, opset).
- ``backbone.onnx`` — per-frame embedding with a dynamic batch axis:
  ``patches[M, 3, P, P]`` → ``features[M, D]``.
- ``head.onnx`` — the temporal head with a dynamic batch axis:
  ``features[N, max_frames, D]`` + ``mask[N, max_frames]`` → ``logit[N]``.
- ``config.yaml`` and ``logistic_calibrator.json`` — copied verbatim from the
  source package.

Splitting the classifier lets the runtime embed only real (unmasked) slots and
cache embeddings per frame across calls; see
``docs/specs/2026-10-14-onnx-embedding-cache-design.md``.

Requires the ``torch`` extra (``torch``, ``timm``, ``onnx``, ``onnxscript``).
Design: ``docs/specs/2026-09-09-onnx-export-design.md``.
"""

import argparse
import copy
import hashlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import yaml

from .onnx_model import (
    BACKBONE_ONNX_FILENAME,
    HEAD_ONNX_FILENAME,
    INPUT_FEATURES,
    INPUT_MASK,
    INPUT_PATCHES,
    ONNX_FORMAT_VERSION,
    OUTPUT_FEATURES,
    OUTPUT_LOGIT,
    score_split,
)
from .onnx_model import (
    MANIFEST_FILENAME as ONNX_MANIFEST_FILENAME,
)
from .package import (
    CONFIG_FILENAME,
    LOGISTIC_CALIBRATOR_FILENAME,
    MANIFEST_FILENAME,
    load_model_package,
)

__all__ = ["export_classifier", "verify_export", "build_onnx_package", "main"]

OPSET_VERSION = 18
DEFAULT_ATOL = 1e-4

# Mask patterns exercised by the parity check: all real, padded tail, a single
# real frame, and interleaved gaps. Every pattern keeps slot 0 real.
_MASK_PATTERNS = ("all", "tail_padded", "single", "interleaved")


def _file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _mask_pattern(name: str, max_frames: int) -> np.ndarray:
    mask = np.ones(max_frames, dtype=bool)
    if name == "tail_padded":
        mask[max(1, max_frames // 2) :] = False
    elif name == "single":
        mask[1:] = False
    elif name == "interleaved":
        mask[1::2] = False
    return mask


def _io_spec(max_frames: int, patch_size: int, feat_dim: int) -> dict[str, Any]:
    return {
        "opset": OPSET_VERSION,
        "backbone": {
            "inputs": {
                INPUT_PATCHES: {
                    "shape": ["batch", 3, patch_size, patch_size],
                    "dtype": "float32",
                }
            },
            "outputs": {
                OUTPUT_FEATURES: {"shape": ["batch", feat_dim], "dtype": "float32"}
            },
        },
        "head": {
            "inputs": {
                INPUT_FEATURES: {
                    "shape": ["batch", max_frames, feat_dim],
                    "dtype": "float32",
                },
                INPUT_MASK: {"shape": ["batch", max_frames], "dtype": "bool"},
            },
            "outputs": {OUTPUT_LOGIT: {"shape": ["batch"], "dtype": "float32"}},
        },
    }


def _export(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        args,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET_VERSION,
        dynamo=True,
        # Positional (one spec per arg): keyed specs must match the forward's own
        # parameter names, which differ between the backbone (x) and head (feats).
        dynamic_shapes=tuple({0: "batch"} for _ in args),
        external_data=False,  # single self-contained file, loadable from bytes
    )
    onnx.checker.check_model(str(output_path))


def export_classifier(
    classifier: torch.nn.Module,
    backbone_path: Path,
    head_path: Path,
    *,
    max_frames: int,
    patch_size: int,
) -> dict[str, Any]:
    """Export ``classifier.backbone`` and ``classifier.head``; return the I/O spec.

    Both graphs get a dynamic batch axis (the backbone batch is "slots to
    embed", the head batch is "tubes"); the frame/patch/feature axes are
    fixed. Exports a deep copy on CPU in eval mode with fused attention
    disabled (timm's fused kernels and the ``nn.TransformerEncoder`` fast
    path do not trace portably) — the caller's module is never touched. Runs
    the ONNX checker on both results.
    """
    classifier = copy.deepcopy(classifier).cpu().eval()
    for module in classifier.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    feat_dim = int(classifier.backbone.feat_dim)

    # Batch-2 example inputs: a batch-1 example lets the tracer fold the size-1
    # batch dim into reshapes, specializing the graph so batch > 1 fails at run.
    _export(
        classifier.backbone,
        (torch.zeros(2, 3, patch_size, patch_size),),
        backbone_path,
        input_names=[INPUT_PATCHES],
        output_names=[OUTPUT_FEATURES],
    )
    _export(
        classifier.head,
        (
            torch.zeros(2, max_frames, feat_dim),
            torch.ones(2, max_frames, dtype=torch.bool),
        ),
        head_path,
        input_names=[INPUT_FEATURES, INPUT_MASK],
        output_names=[OUTPUT_LOGIT],
    )
    return _io_spec(max_frames, patch_size, feat_dim)


def verify_export(
    classifier: torch.nn.Module,
    backbone_path: Path,
    head_path: Path,
    *,
    max_frames: int,
    patch_size: int,
    atol: float = DEFAULT_ATOL,
    seed: int = 0,
) -> float:
    """Compare torch vs onnxruntime logits on random patches; return the max abs diff.

    The torch reference is a deep copy in its as-served configuration (fused
    attention untouched, every slot through the backbone), so parity is
    measured against the production torch path — and the runtime's shortcut
    of embedding only unmasked slots is checked along the way. Each mask
    pattern is checked at batch 1, then all patterns together as one batch
    to exercise the dynamic batch axis.

    Raises:
        ValueError: if any mask pattern (or the batched pass) disagrees by
            more than ``atol``.
    """
    import onnxruntime as ort  # noqa: PLC0415  # runtime-only dependency

    providers = ["CPUExecutionProvider"]
    backbone = ort.InferenceSession(str(backbone_path), providers=providers)
    head = ort.InferenceSession(str(head_path), providers=providers)
    classifier = copy.deepcopy(classifier).cpu().eval()
    rng = np.random.default_rng(seed)
    xs, ms = [], []
    worst = 0.0
    for name in _MASK_PATTERNS:
        x = rng.standard_normal((1, max_frames, 3, patch_size, patch_size)).astype(
            np.float32
        )
        m = _mask_pattern(name, max_frames)[None]
        xs.append(x)
        ms.append(m)
        with torch.no_grad():
            ref = classifier(torch.from_numpy(x), torch.from_numpy(m)).numpy()
        got = score_split(backbone, head, x, m)
        diff = float(np.abs(ref - got).max())
        worst = max(worst, diff)
        if diff > atol:
            raise ValueError(
                f"ONNX parity failed on mask pattern {name!r}: "
                f"|torch - onnx| = {diff:.3g} > atol {atol:.3g}"
            )

    x_batch, m_batch = np.concatenate(xs), np.concatenate(ms)
    with torch.no_grad():
        ref = classifier(torch.from_numpy(x_batch), torch.from_numpy(m_batch)).numpy()
    got = score_split(backbone, head, x_batch, m_batch)
    diff = float(np.abs(ref - got).max())
    worst = max(worst, diff)
    if diff > atol:
        raise ValueError(
            f"ONNX parity failed on the batched pass: "
            f"|torch - onnx| = {diff:.3g} > atol {atol:.3g}"
        )
    return worst


def build_onnx_package(
    model_zip: Path,
    output_path: Path,
    *,
    atol: float = DEFAULT_ATOL,
    allow_uncalibrated: bool = False,
) -> Path:
    """Derive ``model_onnx.zip`` from a ``model.zip``; verify parity along the way.

    Raises:
        UncalibratedModelError: if ``model_zip`` is uncalibrated and
            ``allow_uncalibrated`` is False — refusing at export keeps an
            uncalibrated artifact from being published under an immutable tag
            and only failing on the edge device.
    """
    pkg = load_model_package(
        model_zip, allow_uncalibrated=allow_uncalibrated, with_detector=False
    )
    max_frames = int(pkg.classifier_cfg["max_frames"])
    patch_size = int(pkg.model_input["patch_size"])

    with zipfile.ZipFile(model_zip) as zf:
        src_manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        config_bytes = zf.read(src_manifest["config"])
        calibrator_name = src_manifest.get("logistic_calibrator")
        calibrator_bytes = zf.read(calibrator_name) if calibrator_name else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backbone_path = output_path.with_suffix(".backbone.onnx.tmp")
    head_path = output_path.with_suffix(".head.onnx.tmp")
    try:
        io_spec = export_classifier(
            pkg.classifier,
            backbone_path,
            head_path,
            max_frames=max_frames,
            patch_size=patch_size,
        )
        io_spec["max_abs_logit_diff"] = verify_export(
            pkg.classifier,
            backbone_path,
            head_path,
            max_frames=max_frames,
            patch_size=patch_size,
            atol=atol,
        )

        manifest: dict[str, Any] = {
            "format_version": ONNX_FORMAT_VERSION,
            "backbone_onnx": BACKBONE_ONNX_FILENAME,
            "head_onnx": HEAD_ONNX_FILENAME,
            "config": CONFIG_FILENAME,
            "source": {
                "package": model_zip.name,
                "sha256": _file_sha256(model_zip),
            },
            "onnx": io_spec,
        }
        for key in ("model_version", "variant", "provenance"):
            if key in src_manifest:
                manifest[key] = src_manifest[key]
        if calibrator_bytes is not None:
            manifest["logistic_calibrator"] = LOGISTIC_CALIBRATOR_FILENAME

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(
                ONNX_MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False)
            )
            zf.write(backbone_path, BACKBONE_ONNX_FILENAME)
            zf.write(head_path, HEAD_ONNX_FILENAME)
            zf.writestr(CONFIG_FILENAME, config_bytes)
            if calibrator_bytes is not None:
                zf.writestr(LOGISTIC_CALIBRATOR_FILENAME, calibrator_bytes)
    finally:
        backbone_path.unlink(missing_ok=True)
        head_path.unlink(missing_ok=True)
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="source model.zip")
    parser.add_argument(
        "--output", type=Path, required=True, help="destination model_onnx.zip"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="max |torch - onnx| logit difference tolerated by the parity check",
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="export even if the source model.zip is not calibrated",
    )
    args = parser.parse_args()
    out = build_onnx_package(
        args.model,
        args.output,
        atol=args.atol,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    print(f"exported {args.model} -> {out}")


if __name__ == "__main__":
    main()
