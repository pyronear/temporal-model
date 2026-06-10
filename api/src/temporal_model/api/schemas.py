"""Public request/response DTOs and the mapper from the core model output.

The default response is the lean verdict; ``?verbose=true`` adds a ``details``
block. ``details`` is only set when verbose, so the route serializes with
``exclude_unset=True`` to omit it otherwise (while keeping explicit ``null``s).
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# DNS-style S3 bucket naming: 3-63 chars, lowercase alphanumerics, dots and
# hyphens, starting/ending alphanumeric. Rejects whitespace, uppercase, ARNs
# (colons), slashes and other malformed names early as a 400 instead of letting
# them fail deep in boto3.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class PredictRequest(BaseModel):
    frames: list[str]
    # Optional per-request S3 bucket. Falls back to settings.s3_bucket when
    # omitted (alert-api stacks use per-org dynamic bucket names that no single
    # setting can cover).
    bucket: str | None = None
    # Optional region of interest as normalized corners
    # (x_min, y_min, x_max, y_max) — ultralytics xyxyn convention, suffixed to
    # disambiguate from the xywhn bboxes in responses. Tubes with no real
    # detection intersecting it are dropped before scoring (see
    # docs/specs/2026-06-10-api-roi-design.md).
    roi_xyxyn: tuple[float, float, float, float] | None = None

    @field_validator("frames")
    @classmethod
    def _validate_frames(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("frames must contain at least one S3 key")
        for key in v:
            if "://" in key:
                raise ValueError(f"frame key must be a bare S3 key, not a URL: {key!r}")
        return v

    @field_validator("bucket")
    @classmethod
    def _validate_bucket(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v:
            raise ValueError("bucket must not be empty")
        if not _BUCKET_RE.match(v) or ".." in v:
            raise ValueError(f"bucket is not a valid S3 bucket name: {v!r}")
        return v

    @field_validator("roi_xyxyn")
    @classmethod
    def _validate_roi(
        cls, v: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if v is None:
            return v
        x_min, y_min, x_max, y_max = v
        if not all(0.0 <= c <= 1.0 for c in v):
            raise ValueError("roi_xyxyn coordinates must be in [0, 1]")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("roi_xyxyn requires x_min < x_max and y_min < y_max")
        return v


class FrameEntry(BaseModel):
    frame_idx: int
    bbox: tuple[float, float, float, float] | None = Field(
        description=(
            "Detection box as (cx, cy, w, h) normalized to [0, 1] "
            "(YOLO xywhn convention); null on a gap frame."
        ),
    )
    is_gap: bool
    confidence: float | None


class Tube(BaseModel):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    entries: list[FrameEntry]


class Decision(BaseModel):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float
    threshold_overridden: bool = False
    packaged_threshold: float | None = None


class Preprocessing(BaseModel):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]
    num_tube_candidates: int
    num_tubes_outside_roi: int = 0


class Details(BaseModel):
    decision: Decision
    preprocessing: Preprocessing
    tubes: list[Tube]
    profiling: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    name: str
    version: str | None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    is_smoke: bool
    probability: float | None
    model: ModelInfo
    details: Details | None = None


def _decision_probability(details: dict[str, Any], calibrated: bool) -> float | None:
    """Top-level probability per the API contract.

    None if uncalibrated. Otherwise the max kept-tube probability (0.0 when no
    tubes were kept), regardless of the smoke decision.
    """
    if not calibrated:
        return None
    kept = details["tubes"]["kept"]
    probs = [t["probability"] for t in kept if t.get("probability") is not None]
    return max(probs) if probs else 0.0


def _to_details(
    details: dict[str, Any],
    *,
    threshold_overridden: bool,
    packaged_threshold: float | None,
    profiling: dict[str, Any] | None = None,
) -> Details:
    tubes_block = details["tubes"]
    pre = details["preprocessing"]
    return Details(
        decision=Decision(
            **details["decision"],
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
        ),
        preprocessing=Preprocessing(
            num_frames_input=pre["num_frames_input"],
            num_truncated=pre["num_truncated"],
            padded_frame_indices=pre["padded_frame_indices"],
            num_tube_candidates=tubes_block["num_candidates"],
            num_tubes_outside_roi=tubes_block.get("num_outside_roi", 0),
        ),
        tubes=[Tube(**t) for t in tubes_block["kept"]],
        profiling=profiling,
    )


def to_response(
    out: Any,
    *,
    name: str,
    version: str | None,
    calibrated: bool,
    verbose: bool,
    threshold_overridden: bool = False,
    packaged_threshold: float | None = None,
    profiling: dict[str, Any] | None = None,
) -> PredictResponse:
    """Reshape a core model output into the public response DTO."""
    kwargs: dict[str, Any] = {
        "is_smoke": out.is_positive,
        "probability": _decision_probability(out.details, calibrated),
        "model": ModelInfo(name=name, version=version),
    }
    if verbose:
        kwargs["details"] = _to_details(
            out.details,
            threshold_overridden=threshold_overridden,
            packaged_threshold=packaged_threshold,
            profiling=profiling,
        )
    return PredictResponse(**kwargs)
