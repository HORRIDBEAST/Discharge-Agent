"""
Central configuration via Pydantic Settings.
All values are overridable via environment variables or .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.0

    # Agent loop safety caps
    agent_max_iterations: int = 15
    agent_max_tool_retries: int = 3
    agent_timeout_seconds: int = 120

    # Tracing / observability
    trace_dir: str = "./traces"
    trace_console: bool = True
    trace_file: bool = True

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Safety thresholds
    escalation_confidence_threshold: float = 0.7
    conflict_auto_escalate: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
