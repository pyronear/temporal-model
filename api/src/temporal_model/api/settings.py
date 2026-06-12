"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_API_",
        protected_namespaces=(),
    )

    model_path: str = "/models/model.zip"
    device: str | None = None

    # Overrides the packaged calibrator (logistic) decision threshold when set;
    # out-of-range values fail at startup. Ignored for uncalibrated packages.
    calibrator_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    # Per-frame detection LRU capacity (frame_id → detections). 0 disables.
    detection_cache_size: int = 4096

    # When true, record per-stage timing + cache counts for each request and
    # surface them in logs and the verbose response (`details.profiling`).
    profile: bool = False

    # Shared bearer token guarding POST /predict (env TEMPORAL_API_TOKEN).
    # Unset/empty disables auth (open /predict) — a startup log line reports
    # which mode is active.
    token: str | None = None

    # Release version of the serving code, stamped into the Docker image from
    # the git tag (env TEMPORAL_API_VERSION via a build arg). None on
    # non-release builds → surfaced as null in responses. The alias avoids the
    # env_prefix doubling ("TEMPORAL_API_API_VERSION").
    api_version: str | None = Field(
        default=None, validation_alias="TEMPORAL_API_VERSION"
    )

    s3_bucket: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None

    # Where /predict frames come from when a request omits its `source` field:
    # "s3" downloads keys from a bucket; "local" resolves relative paths under
    # `frames_root` (see docs/specs/2026-06-11-api-local-frames-design.md).
    frame_source: Literal["s3", "local"] = "s3"

    # Root directory for local frames. Required when serving local frames.
    # Settings-only by design — a request-supplied root would let callers
    # probe arbitrary server paths.
    frames_root: str = ""

    @model_validator(mode="after")
    def _require_frames_root_for_local(self) -> "Settings":
        # A local-default server without a root would 400 on every request;
        # fail at boot like other server-level misconfig. (A per-request
        # `source: "local"` override on an s3-default server is still checked
        # in the route — it cannot be known at startup.)
        if self.frame_source == "local" and not self.frames_root:
            raise ValueError(
                "TEMPORAL_API_FRAME_SOURCE=local requires TEMPORAL_API_FRAMES_ROOT"
            )
        return self

    @field_validator("api_version")
    @classmethod
    def _empty_api_version_is_none(cls, v: str | None) -> str | None:
        # ENV TEMPORAL_API_VERSION=${VERSION} with an absent build arg sets "".
        return v or None


settings = Settings()
