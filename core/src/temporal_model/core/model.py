"""TemporalModel implementation for the bbox-tube temporal smoke classifier.

Wires the YOLO companion + the trained torch classifier into the shared
:class:`~temporal_model.core.pipeline.TubePipelineModel` pipeline.
"""

from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from .inference import run_yolo_on_frames
from .logistic_calibrator import LogisticCalibrator
from .package import DEFAULT_AGGREGATION, ModelPackage, load_model_package
from .pipeline import DEFAULT_LOGISTIC_THRESHOLD, TubePipelineModel
from .protocol import Frame
from .types import FrameDetections, Tube
from .yolo_features import YoloFeatureExtractor, model_inputs

__all__ = [
    "BboxTubeTemporalModel",
    "select_device",
    "DEFAULT_AGGREGATION",
    "DEFAULT_LOGISTIC_THRESHOLD",
]


def select_device(device: str | torch.device | None) -> torch.device:
    """Resolve the requested device, auto-picking the best available when None.

    Preference order: CUDA > MPS (Apple Silicon) > CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BboxTubeTemporalModel(TubePipelineModel):
    """YOLO companion + torch tube classifier.

    See ``docs/specs/2026-04-15-temporal-model-protocol-design.md`` for the
    full pipeline description.
    """

    def __init__(
        self,
        *,
        yolo_model: Any,
        classifier: Any,
        config: dict[str, Any],
        device: str | torch.device | None = None,
        calibrator: LogisticCalibrator | None = None,
    ) -> None:
        super().__init__(config=config, calibrator=calibrator)
        self._yolo = yolo_model
        self._device = select_device(device)
        self._classifier = classifier.to(self._device).eval()
        self._yolo_features: YoloFeatureExtractor | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_package(
        cls,
        package_path: Path,
        *,
        device: str | torch.device | None = None,
        allow_uncalibrated: bool = False,
    ) -> Self:
        pkg: ModelPackage = load_model_package(
            package_path, allow_uncalibrated=allow_uncalibrated
        )
        return cls(
            yolo_model=pkg.yolo_model,
            classifier=pkg.classifier,
            config=pkg.config,
            device=device,
            calibrator=pkg.calibrator,
        )

    @classmethod
    def from_archive(
        cls,
        archive_path: Path,
        *,
        device: str | torch.device | None = None,
        allow_uncalibrated: bool = False,
    ) -> Self:
        """Alias for :meth:`from_package`.

        Convenience name used by the evaluation driver so callers can
        refer to the archive by a generic name independent of the internal
        packaging terminology.
        """
        return cls.from_package(
            archive_path, device=device, allow_uncalibrated=allow_uncalibrated
        )

    def detect(self, frames: list[Frame]) -> list[FrameDetections]:
        """Run the companion YOLO detector over ``frames`` (one batched call).

        Pure: same input → same output. Exposed so a serving layer can cache
        per-frame detections and avoid re-detecting frames it has already seen.
        """
        infer = self._cfg["infer"]
        return run_yolo_on_frames(
            self._yolo,
            frames,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )

    def _tube_extras(
        self, tubes: list[Tube], frames: list[Frame]
    ) -> dict[str, np.ndarray]:
        """YOLO neck embeddings per tube entry, when the classifier uses them."""
        if not getattr(self._classifier, "uses_yolo", False):
            return {}
        if self._yolo_features is None:
            self._yolo_features = YoloFeatureExtractor(
                self._yolo,
                image_size=self._cfg["infer"]["image_size"],
                device=self._device,
            )
        max_frames = self._cfg["classifier"]["max_frames"]
        cache: dict = {}
        rois, ctxs = [], []
        for tube in tubes:
            entries = [
                (
                    frames[e.frame_idx].image_path,
                    # Detection-less slots are masked out; any box will do.
                    (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                    if e.detection is not None
                    else (0.5, 0.5, 1.0, 1.0),
                )
                for e in tube.entries[:max_frames]
            ]
            roi, ctx = model_inputs(
                self._yolo_features.tube_features(entries, cache),
                levels=self._classifier.yolo_levels,
                max_frames=max_frames,
            )
            rois.append(roi)
            ctxs.append(ctx)
        return {"yolo_roi": np.stack(rois), "yolo_ctx": np.stack(ctxs)}

    def _score(
        self, patches: np.ndarray, mask: np.ndarray, **extras: np.ndarray
    ) -> np.ndarray:
        """One batched classifier forward over all tubes."""
        p = torch.from_numpy(patches).to(self._device)
        m = torch.from_numpy(mask).to(self._device)
        kw = {k: torch.from_numpy(v).to(self._device) for k, v in extras.items()}
        with torch.no_grad():
            return self._classifier(p, m, **kw).cpu().numpy()
