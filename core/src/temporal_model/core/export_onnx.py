"""Export the torch tube classifier to ONNX and bundle it as ``model_onnx.zip``.

The ONNX package is the torch-free runtime artifact consumed by
:class:`~temporal_model.core.onnx_model.OnnxTemporalModel`. It is derived from
a released ``model.zip`` and published next to it under the same HuggingFace
tag. Contents:

- ``manifest.yaml`` — format version, file pointers, source ``model.zip``
  identity (SHA-256) and the ONNX I/O contract (names, shapes, dtypes, opset).
- ``classifier.onnx`` — the classifier with a dynamic batch axis:
  ``patches[N, max_frames, 3, P, P]`` + ``mask[N, max_frames]`` → ``logit[N]``.
- ``config.yaml`` and ``logistic_calibrator.json`` — copied verbatim from the
  source package.

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
    CLASSIFIER_ONNX_FILENAME,
    INPUT_MASK,
    INPUT_PATCHES,
    ONNX_FORMAT_VERSION,
    OUTPUT_LOGIT,
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


def _io_spec(max_frames: int, patch_size: int) -> dict[str, Any]:
    return {
        "opset": OPSET_VERSION,
        "inputs": {
            INPUT_PATCHES: {
                "shape": ["batch", max_frames, 3, patch_size, patch_size],
                "dtype": "float32",
            },
            INPUT_MASK: {"shape": ["batch", max_frames], "dtype": "bool"},
        },
        "outputs": {OUTPUT_LOGIT: {"shape": ["batch"], "dtype": "float32"}},
    }


def export_classifier(
    classifier: torch.nn.Module,
    output_path: Path,
    *,
    max_frames: int,
    patch_size: int,
) -> dict[str, Any]:
    """Export ``classifier`` to ``output_path``; return the I/O spec.

    The batch axis is dynamic (one ``session.run`` scores all tubes); the
    frame/patch axes are fixed. Exports a deep copy on CPU in eval mode with
    fused attention disabled (timm's fused kernels and the
    ``nn.TransformerEncoder`` fast path do not trace portably) — the caller's
    module is never touched. Runs the ONNX checker on the result.
    """
    classifier = copy.deepcopy(classifier).cpu().eval()
    for module in classifier.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False

    # Batch-2 example input: a batch-1 example lets the tracer fold the size-1
    # batch dim into reshapes, specializing the graph so batch > 1 fails at run.
    patches = torch.zeros(2, max_frames, 3, patch_size, patch_size)
    mask = torch.ones(2, max_frames, dtype=torch.bool)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        classifier,
        (patches, mask),
        str(output_path),
        input_names=[INPUT_PATCHES, INPUT_MASK],
        output_names=[OUTPUT_LOGIT],
        opset_version=OPSET_VERSION,
        dynamo=True,
        dynamic_shapes={INPUT_PATCHES: {0: "batch"}, INPUT_MASK: {0: "batch"}},
        external_data=False,  # single self-contained file, loadable from bytes
    )
    onnx.checker.check_model(str(output_path))
    return _io_spec(max_frames, patch_size)


def verify_export(
    classifier: torch.nn.Module,
    onnx_path: Path,
    *,
    max_frames: int,
    patch_size: int,
    atol: float = DEFAULT_ATOL,
    seed: int = 0,
) -> float:
    """Compare torch vs onnxruntime logits on random patches; return the max abs diff.

    The torch reference is a deep copy in its as-served configuration (fused
    attention untouched), so parity is measured against the production torch
    path, not the export-friendly variant. Each mask pattern is checked at
    batch 1, then all patterns together as one batch to exercise the dynamic
    batch axis.

    Raises:
        ValueError: if any mask pattern (or the batched pass) disagrees by
            more than ``atol``.
    """
    import onnxruntime as ort  # noqa: PLC0415  # runtime-only dependency

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
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
        got = session.run([OUTPUT_LOGIT], {INPUT_PATCHES: x, INPUT_MASK: m})[0]
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
    got = session.run([OUTPUT_LOGIT], {INPUT_PATCHES: x_batch, INPUT_MASK: m_batch})[0]
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
    onnx_path = output_path.with_suffix(".onnx.tmp")
    try:
        io_spec = export_classifier(
            pkg.classifier, onnx_path, max_frames=max_frames, patch_size=patch_size
        )
        io_spec["max_abs_logit_diff"] = verify_export(
            pkg.classifier,
            onnx_path,
            max_frames=max_frames,
            patch_size=patch_size,
            atol=atol,
        )

        manifest: dict[str, Any] = {
            "format_version": ONNX_FORMAT_VERSION,
            "classifier_onnx": CLASSIFIER_ONNX_FILENAME,
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
            zf.write(onnx_path, CLASSIFIER_ONNX_FILENAME)
            zf.writestr(CONFIG_FILENAME, config_bytes)
            if calibrator_bytes is not None:
                zf.writestr(LOGISTIC_CALIBRATOR_FILENAME, calibrator_bytes)
    finally:
        onnx_path.unlink(missing_ok=True)
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
