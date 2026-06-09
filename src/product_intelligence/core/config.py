"""Typed, environment-driven application configuration.

Uses ``pydantic-settings`` so every knob is validated, documented and override-
able via environment variables or a local ``.env`` file. Importing ``settings``
anywhere returns a process-wide singleton.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from product_intelligence import __version__

ModelType = Literal["logreg", "hist_gbdt", "xgboost"]
CalibrationMethod = Literal["sigmoid", "isotonic", "none"]


class Settings(BaseSettings):
    """Central configuration object.

    All fields can be set through environment variables (case-insensitive) or a
    ``.env`` file. See ``.env.example`` for the full surface.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- service metadata ---
    app_name: str = "Pulse360 Product Intelligence API"
    app_version: str = __version__
    environment: Literal["dev", "staging", "prod"] = "dev"

    # --- artifacts / registry ---
    model_artifact_dir: Path = Path("artifacts")
    registry_index_name: str = "registry.json"

    # --- modelling ---
    random_seed: int = 42
    churn_model_type: ModelType = "hist_gbdt"
    churn_calibration: CalibrationMethod = "isotonic"
    recommendation_model_type: Literal["random_forest", "hist_gbdt"] = "random_forest"
    enable_shap: bool = True

    # --- copilot ---
    copilot_use_mock: bool = True

    # --- logging / observability ---
    log_level: str = "INFO"
    log_json: bool = True
    request_id_header: str = "X-Request-ID"
    enable_metrics: bool = True

    # --- API security / limits ---
    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_batch_size: int = 500

    # --- data source (extension: warehouse extracts) ---
    data_source: Literal["synthetic", "file", "sql"] = "synthetic"
    data_path: str | None = None
    data_sql_path: str | None = None
    data_sql_query: str = "SELECT * FROM accounts"

    # --- feature store (extension) ---
    feature_store_dir: Path = Path("feature_store")

    # --- experimentation + feedback (extension) ---
    experiment_salt: str = "pulse360"
    feedback_log_path: Path = Path("feedback/feedback_log.jsonl")

    # --- LLM copilot provider (extension) ---
    llm_provider: Literal["mock", "openai", "anthropic"] = "mock"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 20.0

    # --- rollout: shadow + canary (extension) ---
    shadow_enabled: bool = False
    canary_percent: float = 0.0
    canary_salt: str = "pulse360-canary"

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @property
    def auth_enabled(self) -> bool:
        return self.api_key is not None and self.api_key != ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
