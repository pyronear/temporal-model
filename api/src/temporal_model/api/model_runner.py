"""Model lifecycle and serialized inference.

Loads a packaged ``model.zip`` once, reads its manifest for display metadata,
and runs ``predict_sequence`` off the event loop behind a lock (GPU inference is
not reentrant). The concrete model class lives in ``temporal_model.core`` and is
imported lazily so this module loads while ``core`` is still being migrated.
"""

import asyncio
import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml
from starlette.concurrency import run_in_threadpool

from .detection_cache import DetectionCache

logger = logging.getLogger(__name__)


def read_manifest(package_path: Path) -> dict[str, Any]:
    """Read display metadata from the package manifest.

    Returns ``{"name", "version", "calibrated"}``. ``version`` is ``None`` for
    legacy packages without a ``model_version`` field.
    """
    with zipfile.ZipFile(package_path) as zf:
        manifest = yaml.safe_load(zf.read("manifest.yaml"))
    return {
        "name": manifest.get("variant"),
        "version": manifest.get("model_version"),
        "calibrated": bool(manifest.get("logistic_calibrator")),
    }


def _load_core_model(package_path: Path, device: str | None) -> Any:
    """Lazily import and instantiate the core model from a package."""
    from temporal_model.core.model import BboxTubeTemporalModel  # noqa: PLC0415

    return BboxTubeTemporalModel.from_package(package_path, device=device)


class ModelRunner:
    """Holds the loaded model and serializes inference calls."""

    def __init__(
        self,
        model: Any,
        *,
        name: str,
        version: str | None,
        calibrated: bool,
        threshold_overridden: bool = False,
        packaged_threshold: float | None = None,
        detection_cache_size: int = 0,
    ) -> None:
        self._model = model
        self.name = name
        self.version = version
        self.calibrated = calibrated
        self.threshold_overridden = threshold_overridden
        self.packaged_threshold = packaged_threshold
        self._cache = DetectionCache(detection_cache_size)
        self._lock = asyncio.Lock()

    @classmethod
    def load(
        cls,
        package_path: Path,
        device: str | None,
        calibrator_threshold: float | None = None,
        detection_cache_size: int = 0,
    ) -> "ModelRunner":
        """Load a model package. Call once at startup — this blocks while the
        model checkpoint is deserialized; do not call from a request handler.

        When ``calibrator_threshold`` is set and the model uses the logistic
        decision rule, its threshold is overridden for every prediction. For a
        model that does not decide on the logistic threshold (``max_logit``
        aggregation / uncalibrated) the override has no effect and is ignored
        with a warning — gating on the actual decision rule, not the manifest,
        so the reported override is never a no-op.
        """
        meta = read_manifest(package_path)
        model = _load_core_model(package_path, device)

        threshold_overridden = False
        packaged_threshold = None
        if calibrator_threshold is not None:
            if model.aggregation == "logistic":
                packaged_threshold = model.logistic_threshold
                model.logistic_threshold = calibrator_threshold
                threshold_overridden = True
                logger.info(
                    "calibrator threshold overridden: %s -> %s",
                    packaged_threshold,
                    calibrator_threshold,
                )
            else:
                logger.warning(
                    "TEMPORAL_API_CALIBRATOR_THRESHOLD=%s set but model does not "
                    "use a logistic decision (aggregation=%s); ignoring",
                    calibrator_threshold,
                    model.aggregation,
                )

        return cls(
            model,
            **meta,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
            detection_cache_size=detection_cache_size,
        )

    async def predict(self, frame_paths: list[Path]) -> Any:
        """Resolve detections (cache + detect misses) then run the model.

        The whole orchestration runs in a worker thread under the lock, so the
        cache is accessed by one prediction at a time.
        """
        async with self._lock:
            return await run_in_threadpool(self._predict_sync, frame_paths)

    def _predict_sync(self, frame_paths: list[Path]) -> Any:
        started = time.perf_counter()
        frames = self._model.load_sequence(frame_paths)
        resolved: dict[str, Any] = {}
        misses = []
        for f in frames:
            if f.frame_id in self._cache:
                resolved[f.frame_id] = self._cache.get(f.frame_id)
            else:
                misses.append(f)
        for fd in self._model.detect(misses):
            self._cache.put(fd.frame_id, fd)
            resolved[fd.frame_id] = fd
        out = self._model.predict(frames, frame_detections=resolved)
        logger.info(
            "predict: %d/%d cache hits, seq_len=%d, cache_size=%d, %.0fms",
            len(frames) - len(misses),
            len(frames),
            len(frames),
            len(self._cache),
            (time.perf_counter() - started) * 1000.0,
        )
        return out
