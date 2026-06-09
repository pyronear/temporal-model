"""Core model, tube-building, and inference for the temporal smoke classifier.

Public API. Import the common entry points from here, or reach into the
submodule that owns a symbol (e.g. ``temporal_model.core.tubes``,
``temporal_model.core.model``). The concrete model
(``BboxTubeTemporalModel``) is intentionally NOT re-exported here so that
``import temporal_model.core`` stays light (no torch/timm import); import it
from ``temporal_model.core.model``. Anything prefixed with ``_`` is internal.
"""

from .protocol import Frame, TemporalModel, TemporalModelOutput
from .tubes import build_tubes, merge_colocated_tubes
from .types import Detection, FrameDetections, Tube, TubeEntry

__all__ = [
    "Frame",
    "TemporalModel",
    "TemporalModelOutput",
    "build_tubes",
    "merge_colocated_tubes",
    "Detection",
    "FrameDetections",
    "Tube",
    "TubeEntry",
]
