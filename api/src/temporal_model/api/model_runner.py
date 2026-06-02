"""Model lifecycle and serialized inference.

Loads a packaged ``model.zip`` once, reads its manifest for display metadata,
and runs ``predict_sequence`` off the event loop behind a lock (GPU inference is
not reentrant). The concrete model class lives in ``temporal_model.core`` and is
imported lazily so this module loads while ``core`` is still being migrated.
"""

import asyncio
import zipfile
from pathlib import Path
from typing import Any

import yaml
from starlette.concurrency import run_in_threadpool


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
        self, model: Any, *, name: str, version: str | None, calibrated: bool
    ) -> None:
        self._model = model
        self.name = name
        self.version = version
        self.calibrated = calibrated
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, package_path: Path, device: str | None) -> "ModelRunner":
        """Load a model package. Call once at startup — this blocks while the
        model checkpoint is deserialized; do not call from a request handler."""
        meta = read_manifest(package_path)
        model = _load_core_model(package_path, device)
        return cls(model, **meta)

    async def predict(self, frame_paths: list[Path]) -> Any:
        """Run ``predict_sequence`` in a worker thread, one call at a time."""
        async with self._lock:
            return await run_in_threadpool(self._model.predict_sequence, frame_paths)
