"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from pydantic import Field
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

    s3_bucket: str = ""
    s3_region: str | None = None
    s3_endpoint_url: str | None = None

    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
