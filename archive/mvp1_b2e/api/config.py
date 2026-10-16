from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://pact:pact@localhost:5432/pact_enterprise"

    storage_backend: str = "local"
    storage_local_path: str = "./data/uploads"

    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_research_model: str = "claude-sonnet-4-6"
    llm_timeout_seconds: int = 120

    company_name: str = "Pact"

    max_upload_bytes: int = 26_214_400   # 25 MiB
    max_workspace_words: int = 250_000

    max_job_attempts: int = 3
    job_lease_seconds: int = 120
    job_poll_interval_seconds: int = 5


settings = Settings()
