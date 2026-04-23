"""Configuration loaded from environment (+ optional YAML override).

Every field has a default or is `None`; nothing is required at import time so
tests and scripts can run without a full `.env`. Commands that actually need a
secret (e.g. `serve` in production) raise `ConfigError` when it is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    """Raised when required configuration is missing or malformed."""


Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "development"
    log_level: LogLevel = "INFO"

    s3_bucket: str = Field(default="nutrilens-uploads")
    s3_region: str = Field(default="auto")
    s3_endpoint: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None

    wandb_api_key: SecretStr | None = None
    wandb_project: str = "nutrilens"
    wandb_entity: str | None = None

    model_registry_prefix: str = "models/"

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    serve_host: str = "0.0.0.0"
    serve_port: int = 8000
    serve_shared_secret: SecretStr | None = None


def load_settings(yaml_override: Path | None = None) -> Settings:
    """Load settings from env / `.env`, then optionally overlay a YAML file.

    YAML keys must match the env-var names (case-insensitive). Use this for
    run-specific overrides without polluting the shared `.env`.
    """
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigError(f"invalid settings: {exc}") from exc

    if yaml_override is not None:
        if not yaml_override.is_file():
            raise ConfigError(f"yaml override not found: {yaml_override}")
        overlay = yaml.safe_load(yaml_override.read_text()) or {}
        if not isinstance(overlay, dict):
            raise ConfigError(f"yaml override must be a mapping, got {type(overlay).__name__}")
        try:
            settings = settings.model_copy(update={k.lower(): v for k, v in overlay.items()})
        except ValidationError as exc:
            raise ConfigError(f"invalid yaml override: {exc}") from exc

    return settings


def require(value: SecretStr | str | None, name: str) -> str:
    """Return the value or raise `ConfigError` if missing."""
    if value is None:
        raise ConfigError(f"missing required setting: {name}")
    return value.get_secret_value() if isinstance(value, SecretStr) else value
