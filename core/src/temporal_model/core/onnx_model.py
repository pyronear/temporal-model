"""Torch-free runtime for the tube classifier, backed by onnxruntime.

Loads the ``model_onnx.zip`` built by :mod:`temporal_model.core.export_onnx`
and runs the shared :class:`~temporal_model.core.pipeline.TubePipelineModel`
pipeline with the exported classifier. There is no bundled detector: callers
(e.g. an edge device already running its own YOLO) must pass
``frame_detections`` for every frame.

This module must stay importable with only the ``onnx`` extra installed —
no torch, timm or ultralytics anywhere on its import path.
"""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml

from .logistic_calibrator import LogisticCalibrator
from .pipeline import TubePipelineModel, aggregation_of, require_calibrated

__all__ = [
    "ONNX_MODEL_FILENAME",
    "OnnxPackage",
    "OnnxTemporalModel",
    "load_onnx_package",
]

ONNX_FORMAT_VERSION = 1
ONNX_MODEL_FILENAME = "model_onnx.zip"
MANIFEST_FILENAME = "manifest.yaml"
CLASSIFIER_ONNX_FILENAME = "classifier.onnx"
INPUT_PATCHES = "patches"
INPUT_MASK = "mask"
OUTPUT_LOGIT = "logit"
_CPU_PROVIDERS = ["CPUExecutionProvider"]


@dataclass
class OnnxPackage:
    """A loaded ONNX package: session, config, calibrator and manifest."""

    session: Any  # onnxruntime.InferenceSession
    config: dict[str, Any]
    calibrator: LogisticCalibrator | None
    manifest: dict[str, Any]


def load_onnx_package(
    package_path: Path,
    *,
    allow_uncalibrated: bool = False,
    providers: list[str] | None = None,
) -> OnnxPackage:
    """Load a ``model_onnx.zip`` (read in memory, nothing is extracted).

    Raises:
        FileNotFoundError: if ``package_path`` does not exist.
        KeyError: if the archive is missing expected entries.
        ValueError: if ``format_version`` is unsupported.
        UncalibratedModelError: if the package is uncalibrated and
            ``allow_uncalibrated`` is False.
    """
    import onnxruntime as ort  # noqa: PLC0415  # keep `import temporal_model.core` light

    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path) as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(f"Archive missing {MANIFEST_FILENAME}")
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        version = manifest.get("format_version")
        if version != ONNX_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {version} (expected {ONNX_FORMAT_VERSION})"
            )
        onnx_name = manifest["classifier_onnx"]
        config_name = manifest["config"]
        calibrator_name = manifest.get("logistic_calibrator")
        for n in (onnx_name, config_name, calibrator_name):
            if n is not None and n not in names:
                raise KeyError(f"Archive missing {n}")
        config = yaml.safe_load(zf.read(config_name))
        calibrator = None
        if calibrator_name is not None:
            calibrator = LogisticCalibrator.from_dict(
                json.loads(zf.read(calibrator_name))
            )
            calibrator.verify_sanity_checks()
        model_bytes = zf.read(onnx_name)

    if not allow_uncalibrated:
        require_calibrated(
            calibrator, aggregation_of(config), context="load_onnx_package"
        )

    session = ort.InferenceSession(model_bytes, providers=providers or _CPU_PROVIDERS)
    return OnnxPackage(
        session=session, config=config, calibrator=calibrator, manifest=manifest
    )


class OnnxTemporalModel(TubePipelineModel):
    """Tube classifier served by onnxruntime; detections must be supplied."""

    def __init__(
        self,
        *,
        session: Any,
        config: dict[str, Any],
        calibrator: LogisticCalibrator | None = None,
    ) -> None:
        super().__init__(config=config, calibrator=calibrator)
        self._session = session

    @classmethod
    def from_package(
        cls,
        package_path: Path,
        *,
        allow_uncalibrated: bool = False,
        providers: list[str] | None = None,
    ) -> Self:
        pkg = load_onnx_package(
            package_path, allow_uncalibrated=allow_uncalibrated, providers=providers
        )
        return cls(session=pkg.session, config=pkg.config, calibrator=pkg.calibrator)

    def _score(self, patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Score tubes one at a time: the exported graph has a fixed batch of 1."""
        logits = [
            self._session.run(
                [OUTPUT_LOGIT],
                {
                    INPUT_PATCHES: np.ascontiguousarray(patches[i : i + 1]),
                    INPUT_MASK: np.ascontiguousarray(mask[i : i + 1]),
                },
            )[0][0]
            for i in range(patches.shape[0])
        ]
        return np.asarray(logits, dtype=np.float32)
