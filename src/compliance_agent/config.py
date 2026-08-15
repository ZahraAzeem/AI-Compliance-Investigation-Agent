"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration with local secrets kept out of representations."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Compliance Investigation Agent"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    openai_max_retries: int = Field(default=2, ge=0, le=5)
    openai_max_output_tokens: int = Field(default=1_200, ge=128, le=10_000)
    agent_max_tool_calls: int = Field(default=4, ge=1, le=10)

    @property
    def has_openai_api_key(self) -> bool:
        """Report credential presence without exposing the credential."""

        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""

    return Settings()
