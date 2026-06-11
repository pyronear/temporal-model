"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from pydantic import Field, field_validator
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

    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("api_version")
    @classmethod
    def _empty_api_version_is_none(cls, v: str | None) -> str | None:
        # ENV TEMPORAL_API_VERSION=${VERSION} with an absent build arg sets "".
        return v or None


settings = Settings()
