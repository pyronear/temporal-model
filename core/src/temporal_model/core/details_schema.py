"""Pydantic schema for ``BboxTubeTemporalModel.predict()`` ``details``.

Emitted as a plain ``dict`` by ``model.predict()`` (via ``model_dump()``) to
keep the ``TemporalModelOutput.details: dict`` contract intact.
Consumers that want typed access parse with
``BboxTubeDetails.model_validate(output.details)``.

See ``docs/specs/2026-04-17-details-schema-redesign.md``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "KeptTubeEntry",
    "KeptTube",
    "Preprocessing",
    "Tubes",
    "Decision",
    "BboxTubeDetails",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class KeptTubeEntry(_Frozen):
    frame_idx: int
    bbox: tuple[float, float, float, float] | None
    is_gap: bool
    confidence: float | None


class KeptTube(_Frozen):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None
    entries: list[KeptTubeEntry]


class Preprocessing(_Frozen):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]


class Tubes(_Frozen):
    num_candidates: int
    num_outside_roi: int = 0
    kept: list[KeptTube]


class Decision(_Frozen):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float
    trigger_tube_id: int | None


class BboxTubeDetails(_Frozen):
    preprocessing: Preprocessing
    tubes: Tubes
    decision: Decision
