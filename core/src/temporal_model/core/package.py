"""Model packaging: bundle YOLO weights, classifier checkpoint, and config.

The archive is a standard .zip file containing:

- ``manifest.yaml`` — entry point with format version and file pointers.
- ``yolo_weights.pt`` — ultralytics YOLO checkpoint for the companion detector.
- ``classifier.ckpt`` — Lightning checkpoint for ``TemporalSmokeClassifier``.
- ``config.yaml`` — inference config (infer / tubes / model_input / classifier /
  decision).
"""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .detector import load_detector
from .logistic_calibrator import LogisticCalibrator
from .temporal_classifier import TemporalSmokeClassifier

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
YOLO_WEIGHTS_FILENAME = "yolo_weights.pt"
CLASSIFIER_CKPT_FILENAME = "classifier.ckpt"
CONFIG_FILENAME = "config.yaml"
LOGISTIC_CALIBRATOR_FILENAME = "logistic_calibrator.json"
DEFAULT_EXTRACT_DIR = Path(".cache/temporal_model_core")


@dataclass
class ModelPackage:
    """A loaded model package: classifier, YOLO model, and full config."""

    classifier: Any  # TemporalSmokeClassifier; Any avoids import cycles in this module
    yolo_model: Any  # ultralytics.YOLO; same reason
    config: dict[str, Any]
    calibrator: LogisticCalibrator | None = None

    @property
    def infer(self) -> dict[str, Any]:
        return self.config["infer"]

    @property
    def tubes(self) -> dict[str, Any]:
        return self.config["tubes"]

    @property
    def model_input(self) -> dict[str, Any]:
        return self.config["model_input"]

    @property
    def classifier_cfg(self) -> dict[str, Any]:
        return self.config["classifier"]

    @property
    def decision(self) -> dict[str, Any]:
        return self.config["decision"]


def build_model_package(
    *,
    yolo_weights_path: Path,
    classifier_ckpt_path: Path,
    config: dict[str, Any],
    variant: str,
    output_path: Path,
    model_version: str | None = None,
    train_git_sha: str | None = None,
    calibrator: LogisticCalibrator | None = None,
) -> Path:
    """Bundle YOLO weights + classifier checkpoint + config into a .zip archive.

    Args:
        yolo_weights_path: Path to the ultralytics YOLO ``.pt`` file.
        classifier_ckpt_path: Path to the Lightning ``.ckpt`` for
            ``TemporalSmokeClassifier``.
        config: Full package config dict (see module docstring for schema).
        variant: Identifier recorded in the manifest (informational).
        output_path: Destination ``.zip`` path.
        model_version: Optional released model version, stamped into the
            manifest as ``model_version`` (omitted when ``None``).
        train_git_sha: Optional git SHA of the training code, recorded under
            ``provenance.train_git_sha``.
        calibrator: Optional fitted :class:`LogisticCalibrator`. When
            provided, its JSON payload is bundled into the archive and
            the manifest gets a ``logistic_calibrator`` pointer.

    Returns:
        The resolved ``output_path``.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")
    if not classifier_ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_ckpt_path}"
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
        "config": CONFIG_FILENAME,
    }
    if calibrator is not None:
        manifest["logistic_calibrator"] = LOGISTIC_CALIBRATOR_FILENAME
    if model_version is not None:
        manifest["model_version"] = model_version
    manifest["provenance"] = {
        "train_git_sha": train_git_sha,
        "backbone": config.get("classifier", {}).get("backbone"),
        "detector": load_detector().model_dump(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(yolo_weights_path, YOLO_WEIGHTS_FILENAME)
        zf.write(classifier_ckpt_path, CLASSIFIER_CKPT_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))
        if calibrator is not None:
            payload = {
                "features": list(calibrator.features),
                "coefficients": [float(c) for c in calibrator.coefficients],
                "intercept": float(calibrator.intercept),
                "sanity_checks": list(calibrator.sanity_checks),
            }
            zf.writestr(
                LOGISTIC_CALIBRATOR_FILENAME,
                json.dumps(payload, indent=2),
            )
    return output_path.resolve()


def _load_yolo(weights_path: Path) -> Any:
    """Thin wrapper around ultralytics.YOLO.

    The ``ultralytics`` import is deliberately inside the function body so
    tests can patch ``_load_yolo`` without triggering the heavy import chain.
    This is the one and only sanctioned import-inside-function in this
    project (carries a PLC0415 noqa).
    """
    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(str(weights_path))


def _build_classifier(classifier_cfg: dict[str, Any]) -> TemporalSmokeClassifier:
    """Instantiate a ``TemporalSmokeClassifier`` from a packaged config block.

    Mirrors the classifier kwargs written by ``scripts/package_model.py`` —
    the packaged ``config["classifier"]`` dict carries every kwarg needed to
    reconstruct the exact training-time architecture (including ViT-specific
    ``global_pool``/``img_size`` and transformer-head hyperparameters).
    """
    kwargs: dict[str, Any] = {
        "backbone": classifier_cfg["backbone"],
        "pretrained": classifier_cfg.get("pretrained", False),
        "finetune": classifier_cfg.get("finetune", False),
        "finetune_last_n_blocks": classifier_cfg.get("finetune_last_n_blocks", 0),
        "max_frames": classifier_cfg.get("max_frames", 20),
        "global_pool": classifier_cfg.get("global_pool", "token"),
    }
    for k in (
        "transformer_num_layers",
        "transformer_num_heads",
        "transformer_ffn_dim",
        "transformer_dropout",
        "img_size",
    ):
        if k in classifier_cfg:
            kwargs[k] = classifier_cfg[k]
    return TemporalSmokeClassifier(**kwargs)


def _load_classifier(
    ckpt_path: Path, classifier_cfg: dict[str, Any]
) -> TemporalSmokeClassifier:
    """Build a ``TemporalSmokeClassifier`` from config and load its weights.

    Accepts both Lightning-style ckpts (``{"state_dict": {"model.xxx": ...}}``)
    and plain state_dicts (``{"xxx": ...}``).
    """
    model = _build_classifier(classifier_cfg)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        raw = blob["state_dict"]
        sd = {
            k.removeprefix("model."): v
            for k, v in raw.items()
            if k.startswith("model.")
        }
    else:
        sd = blob
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Args:
        package_path: Path to a ``.zip`` built by :func:`build_model_package`.
        extract_dir: Where to extract YOLO weights and classifier ckpt.

    Raises:
        FileNotFoundError: if ``package_path`` does not exist.
        KeyError: if the archive is missing expected entries.
        ValueError: if ``format_version`` is unsupported.
    """
    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(f"Archive missing {MANIFEST_FILENAME}")
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))

        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {version} (expected {FORMAT_VERSION})"
            )

        yolo_name = manifest["yolo_weights"]
        ckpt_name = manifest["classifier_checkpoint"]
        config_name = manifest["config"]
        for n in (yolo_name, ckpt_name, config_name):
            if n not in names:
                raise KeyError(f"Archive missing {n}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(yolo_name, extract_dir)
        zf.extract(ckpt_name, extract_dir)
        config = yaml.safe_load(zf.read(config_name))

        calibrator: LogisticCalibrator | None = None
        calibrator_name = manifest.get("logistic_calibrator")
        if calibrator_name is not None:
            if calibrator_name not in names:
                raise KeyError(f"Archive missing {calibrator_name}")
            payload = json.loads(zf.read(calibrator_name))
            calibrator = LogisticCalibrator(
                features=list(payload["features"]),
                coefficients=np.asarray(payload["coefficients"], dtype=float),
                intercept=float(payload["intercept"]),
                sanity_checks=list(payload.get("sanity_checks", [])),
            )
            calibrator.verify_sanity_checks()

    yolo_model = _load_yolo(extract_dir / yolo_name)
    classifier = _load_classifier(extract_dir / ckpt_name, config["classifier"])
    return ModelPackage(
        classifier=classifier,
        yolo_model=yolo_model,
        config=config,
        calibrator=calibrator,
    )
