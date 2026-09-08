"""Backend-agnostic tube pipeline: pad → detect → tubes → crop → score → decide.

:class:`TubePipelineModel` owns every stage of ``predict()`` except the two
that need a deep-learning runtime: detection (:meth:`TubePipelineModel.detect`)
and tube scoring (:meth:`TubePipelineModel._score`). Backends subclass it —
:class:`~temporal_model.core.model.BboxTubeTemporalModel` (torch + YOLO) and
:class:`~temporal_model.core.onnx_model.OnnxTemporalModel` (onnxruntime,
supplied detections only). This module must stay importable without torch.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from .details_schema import (
    BboxTubeDetails,
    Decision,
    KeptTube,
    KeptTubeEntry,
    Preprocessing,
    Tubes,
)
from .inference import (
    build_tubes_for_inference,
    crop_tube_patches,
    find_first_crossing_trigger,
    make_decision_fn,
    pad_frames_symmetrically,
    pad_frames_uniform,
)
from .logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
    tube_feature_dict,
)
from .protocol import Frame, TemporalModel, TemporalModelOutput
from .stabilize import tube_stabilized_window
from .stage_timer import StageTimer, stage_ctx
from .tubes import build_tubes, tube_intersects_roi, validate_roi
from .types import FrameDetections

__all__ = [
    "TubePipelineModel",
    "DEFAULT_AGGREGATION",
    "DEFAULT_LOGISTIC_THRESHOLD",
]

_PAD_STRATEGIES = {
    "symmetric": pad_frames_symmetrically,
    "uniform": pad_frames_uniform,
}

DEFAULT_AGGREGATION = "max_logit"
DEFAULT_LOGISTIC_THRESHOLD = 0.5


class TubePipelineModel(TemporalModel):
    """Shared ``predict()`` for the bbox-tube classifier, minus the runtime.

    Subclasses provide :meth:`_score` and, when they bundle a detector,
    :meth:`detect`. See ``docs/specs/2026-04-15-temporal-model-protocol-design.md``
    for the full pipeline description.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        calibrator: LogisticCalibrator | None = None,
    ) -> None:
        self._cfg = config
        self._calibrator = calibrator

    @property
    def aggregation(self) -> str:
        """Decision aggregation rule: ``"max_logit"`` or ``"logistic"``."""
        return self._cfg["decision"].get("aggregation", DEFAULT_AGGREGATION)

    @property
    def logistic_threshold(self) -> float:
        """Probability threshold for the logistic decision rule."""
        return float(
            self._cfg["decision"].get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)
        )

    @logistic_threshold.setter
    def logistic_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"logistic_threshold must be in [0, 1], got {value}")
        self._cfg["decision"]["logistic_threshold"] = float(value)

    def detect(self, frames: list[Frame]) -> list[FrameDetections]:
        """Run the companion detector over ``frames``.

        Backends without a bundled detector leave this unimplemented; their
        ``predict()`` then requires ``frame_detections`` covering every frame.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no detector: pass frame_detections for "
            "every frame"
        )

    def _score(self, patches: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Score tubes: ``patches[N,T,3,H,W] float32, mask[N,T] bool -> logits[N]``."""
        raise NotImplementedError

    def _resolve_frame_detections(
        self,
        truncated: list[Frame],
        frame_detections: dict[str, FrameDetections],
    ) -> list[FrameDetections]:
        """Use supplied detections where present, detect the rest, in order.

        Each entry's positional ``frame_idx`` is re-stamped to its index in
        ``truncated`` — a cached entry carries the ``frame_idx`` from the call
        that produced it, which is meaningless here.
        """
        misses = [f for f in truncated if f.frame_id not in frame_detections]
        fresh = {fd.frame_id: fd for fd in self.detect(misses)} if misses else {}
        resolved: list[FrameDetections] = []
        for idx, f in enumerate(truncated):
            fd = frame_detections.get(f.frame_id)
            if fd is None:
                fd = fresh[f.frame_id]
            resolved.append(replace(fd, frame_idx=idx))
        return resolved

    def predict(
        self,
        frames: list[Frame],
        *,
        frame_detections: dict[str, FrameDetections] | None = None,
        roi: tuple[float, float, float, float] | None = None,
        timer: StageTimer | None = None,
        compute_trigger: bool = False,
    ) -> TemporalModelOutput:
        if roi is not None:
            validate_roi(roi)

        infer = self._cfg["infer"]
        tubes_cfg = self._cfg["tubes"]
        mi = self._cfg["model_input"]
        clf_cfg = self._cfg["classifier"]
        dec = self._cfg["decision"]

        aggregation = dec.get("aggregation", DEFAULT_AGGREGATION)
        effective_threshold = (
            float(dec["logistic_threshold"])
            if aggregation == "logistic"
            else float(dec["threshold"])
        )

        def _make_details(
            *,
            num_frames_input: int,
            num_truncated: int,
            padded_indices: list[int],
            num_candidates: int,
            num_outside_roi: int,
            kept_tubes_models: list[KeptTube],
            trigger_tube_id: int | None,
        ) -> dict:
            return BboxTubeDetails(
                preprocessing=Preprocessing(
                    num_frames_input=num_frames_input,
                    num_truncated=num_truncated,
                    padded_frame_indices=padded_indices,
                ),
                tubes=Tubes(
                    num_candidates=num_candidates,
                    num_outside_roi=num_outside_roi,
                    kept=kept_tubes_models,
                ),
                decision=Decision(
                    aggregation=aggregation,
                    threshold=effective_threshold,
                    trigger_tube_id=trigger_tube_id,
                ),
            ).model_dump()

        original_len = len(frames)
        if original_len == 0:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details=_make_details(
                    num_frames_input=0,
                    num_truncated=0,
                    padded_indices=[],
                    num_candidates=0,
                    num_outside_roi=0,
                    kept_tubes_models=[],
                    trigger_tube_id=None,
                ),
            )

        with stage_ctx(timer, "pad"):
            truncated = frames[: clf_cfg["max_frames"]]
            n_truncated = original_len - len(truncated)

            padded_indices: list[int] = []
            pad_min = int(infer.get("pad_to_min_frames", 0))
            if pad_min > 0 and len(truncated) < pad_min:
                strategy = infer.get("pad_strategy", "symmetric")
                try:
                    pad_fn = _PAD_STRATEGIES[strategy]
                except KeyError as e:
                    raise ValueError(
                        f"unknown pad_strategy {strategy!r}; "
                        f"expected one of {sorted(_PAD_STRATEGIES)}"
                    ) from e
                truncated, padded_indices = pad_fn(truncated, min_length=pad_min)

        if frame_detections is None:
            # Time the "detector" stage only when we actually detect. When
            # detections are supplied (a serving layer that detects + caches
            # upstream), that real cost is timed there; resolving the supplied
            # dict here is negligible and must not be conflated into "detector".
            with stage_ctx(timer, "detector"):
                frame_dets = self.detect(truncated)
        else:
            frame_dets = self._resolve_frame_detections(truncated, frame_detections)

        with stage_ctx(timer, "tubes"):
            # Pre-merge (raw) candidates count, for the details JSON.
            candidate_tubes = build_tubes(
                frame_dets,
                iou_threshold=tubes_cfg["iou_threshold"],
                max_misses=tubes_cfg["max_misses"],
            )
            kept = build_tubes_for_inference(
                frame_dets,
                iou_threshold=tubes_cfg["iou_threshold"],
                max_misses=tubes_cfg["max_misses"],
                min_tube_length=tubes_cfg["infer_min_tube_length"],
                min_detected_entries=tubes_cfg["min_detected_entries"],
                interpolate_gaps=tubes_cfg["interpolate_gaps"],
                merge_iomin=tubes_cfg.get("merge_iomin"),
                merge_prox_factor=tubes_cfg.get("merge_prox_factor"),
                merge_max_gap=tubes_cfg.get("merge_max_gap"),
            )
            num_outside_roi = 0
            if roi is not None:
                n_before = len(kept)
                kept = [t for t in kept if tube_intersects_roi(t, roi)]
                num_outside_roi = n_before - len(kept)

        if not kept:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details=_make_details(
                    num_frames_input=original_len,
                    num_truncated=n_truncated,
                    padded_indices=padded_indices,
                    num_candidates=len(candidate_tubes),
                    num_outside_roi=num_outside_roi,
                    kept_tubes_models=[],
                    trigger_tube_id=None,
                ),
            )

        patches_per_tube: list[np.ndarray] = []
        masks_per_tube: list[np.ndarray] = []
        with stage_ctx(timer, "crop"):
            for t in kept:
                p, m = crop_tube_patches(
                    t,
                    truncated,
                    context_factor=mi["context_factor"],
                    patch_size=mi["patch_size"],
                    max_frames=clf_cfg["max_frames"],
                    normalization_mean=mi["normalization"]["mean"],
                    normalization_std=mi["normalization"]["std"],
                    stabilize=mi.get("stabilize", True),
                )
                patches_per_tube.append(p)
                masks_per_tube.append(m)

        with stage_ctx(timer, "classifier"):
            logits = self._score(
                np.stack(patches_per_tube, axis=0), np.stack(masks_per_tube, axis=0)
            )

        with stage_ctx(timer, "trigger_search"):
            if compute_trigger:
                is_positive, trigger, trigger_tube_id, per_tube_first_crossing = (
                    find_first_crossing_trigger(
                        classifier=self._score,
                        tubes=kept,
                        patches_per_tube=patches_per_tube,
                        masks_per_tube=masks_per_tube,
                        full_logits=logits,
                        aggregation=aggregation,
                        threshold=float(dec["threshold"]),
                        calibrator=self._calibrator,
                        logistic_threshold=float(
                            dec.get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)
                        ),
                        min_prefix_length=tubes_cfg["infer_min_tube_length"],
                    )
                )
            else:
                decides_positive = make_decision_fn(
                    aggregation,
                    threshold=float(dec["threshold"]),
                    calibrator=self._calibrator,
                    logistic_threshold=float(
                        dec.get("logistic_threshold", DEFAULT_LOGISTIC_THRESHOLD)
                    ),
                )
                n_kept = len(kept)
                is_positive = any(
                    decides_positive(float(logits[i]), tube, n_kept)
                    for i, tube in enumerate(kept)
                )
                trigger = None
                trigger_tube_id = None
                per_tube_first_crossing = {}

        logits_list: list[float] = [float(x) for x in logits]

        def _probability_for(tube_idx: int, raw_logit: float) -> float | None:
            if self._calibrator is None:
                return None
            tube_dict = tube_feature_dict(kept[tube_idx], raw_logit)
            features = extract_features(tube_dict, n_tubes=len(kept))
            return float(self._calibrator.predict_proba(features))

        kept_models: list[KeptTube] = []
        for tube_idx, tube in enumerate(kept):
            entries_models = [
                KeptTubeEntry(
                    frame_idx=e.frame_idx,
                    bbox=(
                        (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                        if e.detection is not None
                        else None
                    ),
                    is_gap=e.is_gap,
                    confidence=(
                        e.detection.confidence if e.detection is not None else None
                    ),
                )
                for e in tube.entries
            ]
            first_crossing = per_tube_first_crossing.get(tube.tube_id, {}).get(
                "crossing_frame"
            )
            kept_models.append(
                KeptTube(
                    tube_id=tube.tube_id,
                    start_frame=tube.start_frame,
                    end_frame=tube.end_frame,
                    logit=logits_list[tube_idx],
                    probability=_probability_for(tube_idx, logits_list[tube_idx]),
                    first_crossing_frame=first_crossing,
                    entries=entries_models,
                    stabilized_window=(
                        tube_stabilized_window(tube.entries)
                        if mi.get("stabilize", True)
                        else None
                    ),
                )
            )

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger,
            details=_make_details(
                num_frames_input=original_len,
                num_truncated=n_truncated,
                padded_indices=padded_indices,
                num_candidates=len(candidate_tubes),
                num_outside_roi=num_outside_roi,
                kept_tubes_models=kept_models,
                trigger_tube_id=trigger_tube_id,
            ),
        )
