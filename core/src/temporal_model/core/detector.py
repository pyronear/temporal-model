"""Companion detector: single source of truth + typed accessor.

The bundled YOLO detector identity is declared once in ``detector.yaml`` and
propagated into each packaged model's manifest provenance. It cannot be derived
from training (the training pipeline does not run YOLO), so it is declared here
and verified by SHA-256 against the published HuggingFace weights.

See ``docs/specs/2026-06-03-model-versioning-design.md``.
"""

from importlib.resources import files

import yaml
from pydantic import BaseModel, ConfigDict

__all__ = ["Detector", "load_detector", "DETECTOR_WEIGHTS_FILENAME"]

DETECTOR_YAML_FILENAME = "detector.yaml"
# The weights file published in the HF detector repo (pyronear convention).
DETECTOR_WEIGHTS_FILENAME = "best.pt"
_HF_PREFIX = "hf:"


class Detector(BaseModel):
    """Identity of the companion detector bundled into ``model.zip``."""

    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    source: str
    sha256: str

    @property
    def repo_id(self) -> str:
        """The HF repo id, e.g. ``pyronear/yolo11s_nimble-narwhal_v6.0.0``."""
        if not self.source.startswith(_HF_PREFIX):
            raise ValueError(
                f"Unsupported detector source: {self.source!r} "
                f"(expected '{_HF_PREFIX}<org>/<repo>')"
            )
        return self.source[len(_HF_PREFIX) :]


def load_detector() -> Detector:
    """Read and validate the detector source of truth (``detector.yaml``)."""
    text = (files("temporal_model.core") / DETECTOR_YAML_FILENAME).read_text()
    data = yaml.safe_load(text)
    return Detector.model_validate(data["detector"])
