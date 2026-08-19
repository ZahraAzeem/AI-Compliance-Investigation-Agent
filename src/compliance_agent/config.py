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

    ai_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    ai_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ai_max_retries: int = Field(default=2, ge=0, le=5)
    ai_max_output_tokens: int = Field(default=2_000, ge=128, le=10_000)
    ai_max_output_retries: int = Field(default=1, ge=0, le=2)
    agent_max_tool_calls: int = Field(default=4, ge=1, le=10)
    agent_max_elapsed_seconds: float = Field(default=60.0, gt=0, le=300)

    @property
    def selected_ai_api_key(self) -> SecretStr | None:
        """Select the configured provider credential without revealing it."""

        return self.groq_api_key if self.ai_provider == "groq" else self.openai_api_key

    @property
    def selected_ai_model(self) -> str:
        """Select the model configured for the active provider."""

        return self.groq_model if self.ai_provider == "groq" else self.openai_model

    @property
    def selected_ai_base_url(self) -> str | None:
        """Return a compatibility base URL only when the provider requires one."""

        return self.groq_base_url if self.ai_provider == "groq" else None

    @property
    def has_ai_api_key(self) -> bool:
        """Report selected credential presence without exposing the credential."""

        key = self.selected_ai_api_key
        return bool(key and key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process."""

    return Settings()
