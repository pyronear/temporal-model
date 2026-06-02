"""Runtime configuration for the API, read from ``TEMPORAL_API_*`` env vars."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEMPORAL_API_",
        protected_namespaces=(),
    )

    model_path: str = "/models/model.zip"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
