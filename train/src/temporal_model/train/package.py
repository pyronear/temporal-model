"""Build a deployable, calibrated model.zip for the ViT temporal classifier.

Loads the trained classifier + the DVC-tracked detector, fits a logistic
calibrator in memory (running the full pipeline on train/val), builds the
inference config from params.yaml, and writes model.zip via core. The human
``model_version`` is NOT stamped here — the release ``publish`` step applies it.

Usage:
    uv run python -m temporal_model.train.package \\
        --variant vit_dinov2_finetune \\
        --output data/06_models/vit_dinov2_finetune/model.zip
"""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from temporal_model.core.detector import Detector, load_detector
from temporal_model.core.logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
)
from temporal_model.core.model import BboxTubeTemporalModel
from temporal_model.core.package import _load_yolo, build_model_package
from temporal_model.core.temporal_classifier import TemporalSmokeClassifier
from temporal_model.train.calibration import calibrate_threshold
from temporal_model.train.logistic_calibrator_fit import fit as fit_logistic_calibrator
from temporal_model.train.package_predict import collect_pipeline_records
from temporal_model.train.val_predict import collect_val_probabilities

_NORMALIZATION = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def classifier_kwargs(variant_cfg: dict) -> dict:
    """Transformer-only classifier kwargs (matches core._build_classifier)."""
    kwargs: dict = {
        "backbone": variant_cfg["backbone"],
        "pretrained": False,
        "finetune": variant_cfg.get("finetune", False),
        "finetune_last_n_blocks": variant_cfg.get("finetune_last_n_blocks", 0),
        "max_frames": variant_cfg.get("max_frames", 20),
        "global_pool": variant_cfg.get("global_pool", "token"),
    }
    for k in (
        "transformer_num_layers",
        "transformer_num_heads",
        "transformer_ffn_dim",
        "transformer_dropout",
        "img_size",
    ):
        if k in variant_cfg:
            kwargs[k] = variant_cfg[k]
    return kwargs


def tubes_config(all_params: dict) -> dict:
    tubes_params = all_params["tubes"]
    build_tubes_params = all_params["build_tubes"]
    cfg: dict = {
        "iou_threshold": tubes_params["iou_threshold"],
        "max_misses": tubes_params["max_misses"],
        "min_tube_length": build_tubes_params["min_tube_length"],
        "infer_min_tube_length": all_params["package"]["infer_min_tube_length"],
        "min_detected_entries": build_tubes_params["min_detected_entries"],
        "interpolate_gaps": True,
    }
    merge_iomin = tubes_params.get("merge_iomin")
    merge_prox_factor = tubes_params.get("merge_prox_factor")
    merge_max_gap = tubes_params.get("merge_max_gap")
    if all(v is not None for v in (merge_iomin, merge_prox_factor, merge_max_gap)):
        cfg["merge_iomin"] = float(merge_iomin)
        cfg["merge_prox_factor"] = float(merge_prox_factor)
        cfg["merge_max_gap"] = int(merge_max_gap)
    return cfg


def build_config(
    all_params: dict,
    variant_cfg: dict,
    threshold: float,
    *,
    aggregation: str,
    logistic_threshold: float | None,
) -> dict:
    package_params = all_params["package"]
    decision: dict = {
        "aggregation": aggregation,
        "threshold": float(threshold),
        "target_recall": package_params["target_recall"],
        "trigger_rule": "end_of_winner",
    }
    if logistic_threshold is not None:
        decision["logistic_threshold"] = float(logistic_threshold)
    return {
        "infer": package_params["infer"],
        "tubes": tubes_config(all_params),
        "model_input": {
            "context_factor": all_params["model_input"]["context_factor"],
            "patch_size": all_params["model_input"]["patch_size"],
            "normalization": _NORMALIZATION,
        },
        "classifier": classifier_kwargs(variant_cfg),
        "decision": decision,
    }


def verify_detector_weights(yolo_weights_path: Path, detector: Detector) -> None:
    """Assert the bundled detector weights match the declared ``detector.sha256``.

    Guards provenance integrity: a hand-placed or wrong ``yolo_weights.pt`` would
    otherwise be packaged with a manifest claiming a detector it isn't.

    Raises:
        ValueError: if the file's SHA-256 differs from ``detector.sha256``.
    """
    h = hashlib.sha256()
    with yolo_weights_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != detector.sha256:
        raise ValueError(
            f"Detector weights SHA-256 mismatch for {detector.name}: "
            f"expected {detector.sha256}, got {actual} ({yolo_weights_path})"
        )


def _load_classifier_from_ckpt(
    ckpt_path: Path, variant_cfg: dict
) -> TemporalSmokeClassifier:
    model = TemporalSmokeClassifier(**classifier_kwargs(variant_cfg))
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = (
        blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    )
    sd = {
        k.removeprefix("model."): v for k, v in raw.items() if k.startswith("model.")
    } or raw
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _calibrated_probs(
    records: list[dict], calibrator: LogisticCalibrator
) -> np.ndarray:
    probs = []
    for r in records:
        kept = r["kept_tubes"]
        if not kept:
            probs.append(0.0)
        else:
            best = max(kept, key=lambda t: t["logit"])
            probs.append(
                calibrator.predict_proba(extract_features(best, n_tubes=len(kept)))
            )
    return np.array(probs)


def _labels_array(records: list[dict]) -> np.ndarray:
    return np.array([1 if r["label"] == "smoke" else 0 for r in records])


def _fit_calibrator_and_threshold(
    *,
    yolo_weights_path: Path,
    classifier: TemporalSmokeClassifier,
    pipeline_config: dict,
    raw_train_dir: Path,
    raw_val_dir: Path,
    target_recall: float,
) -> tuple[LogisticCalibrator, float]:
    os.environ.setdefault("YOLO_VERBOSE", "False")  # quiet per-frame YOLO logs
    print("[package] loading detector + building in-memory pipeline...", flush=True)
    fit_model = BboxTubeTemporalModel(
        yolo_model=_load_yolo(yolo_weights_path),
        classifier=classifier,
        config=pipeline_config,
    )
    print(
        "[package] running pipeline on TRAIN sequences (calibrator fit)...", flush=True
    )
    train_records = collect_pipeline_records(model=fit_model, raw_dir=raw_train_dir)
    print(
        f"[package] fitting logistic calibrator on {len(train_records)} records...",
        flush=True,
    )
    calibrator = fit_logistic_calibrator(train_records)
    print("[package] running pipeline on VAL sequences (threshold)...", flush=True)
    val_records = collect_pipeline_records(model=fit_model, raw_dir=raw_val_dir)
    probs = _calibrated_probs(val_records, calibrator)
    labels = _labels_array(val_records)
    logistic_threshold = calibrate_threshold(probs, labels, target_recall=target_recall)
    return calibrator, float(logistic_threshold)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, default=Path("params.yaml"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--yolo-weights-path", type=Path, default=None)
    parser.add_argument(
        "--val-patches-dir", type=Path, default=Path("data/05_model_input/val")
    )
    parser.add_argument(
        "--raw-train-dir", type=Path, default=Path("data/01_raw/datasets/train")
    )
    parser.add_argument(
        "--raw-val-dir", type=Path, default=Path("data/01_raw/datasets/val")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_params: dict[str, Any] = yaml.safe_load(args.params_path.read_text())
    variant_cfg = all_params[f"train_{args.variant}"]
    package_params = all_params["package"]

    detector = load_detector()
    checkpoint = args.checkpoint_path or (
        Path("data/06_models") / args.variant / "best_checkpoint.pt"
    )
    yolo_weights = args.yolo_weights_path or (
        Path("data/06_models/detectors") / detector.name / "yolo_weights.pt"
    )
    print(
        f"[package] detector={detector.name}; verifying weights SHA-256...", flush=True
    )
    verify_detector_weights(yolo_weights, detector)

    print(f"[package] loading classifier from {checkpoint}...", flush=True)
    classifier = _load_classifier_from_ckpt(checkpoint, variant_cfg)
    print("[package] scoring val patches (classifier-only threshold)...", flush=True)
    probs, labels = collect_val_probabilities(
        classifier,
        args.val_patches_dir,
        max_frames=variant_cfg["max_frames"],
        batch_size=variant_cfg.get("batch_size", 32),
        num_workers=variant_cfg.get("num_workers", 4),
    )
    threshold = calibrate_threshold(
        probs, labels, target_recall=package_params["target_recall"]
    )
    print(
        f"[package] val patches scored: n={len(probs)} threshold={threshold:.4f}",
        flush=True,
    )

    aggregation = package_params.get("aggregation", {}).get(args.variant, "max_logit")
    calibrator: LogisticCalibrator | None = None
    logistic_threshold: float | None = None
    if aggregation == "logistic":
        pipeline_config = build_config(
            all_params,
            variant_cfg,
            threshold,
            aggregation="max_logit",
            logistic_threshold=None,
        )
        calibrator, logistic_threshold = _fit_calibrator_and_threshold(
            yolo_weights_path=yolo_weights,
            classifier=classifier,
            pipeline_config=pipeline_config,
            raw_train_dir=args.raw_train_dir,
            raw_val_dir=args.raw_val_dir,
            target_recall=package_params["target_recall"],
        )

    config = build_config(
        all_params,
        variant_cfg,
        threshold,
        aggregation=aggregation,
        logistic_threshold=logistic_threshold,
    )
    build_model_package(
        yolo_weights_path=yolo_weights,
        classifier_ckpt_path=checkpoint,
        config=config,
        variant=args.variant,
        output_path=args.output,
        calibrator=calibrator,
        train_git_sha=_git_sha(),
    )
    print(
        f"[package] wrote {args.output} | variant={args.variant} "
        f"aggregation={aggregation} threshold={threshold:.4f}"
    )


if __name__ == "__main__":
    main()
