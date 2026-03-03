"""
Centralized configuration via environment variables.

Why Pydantic for config? Type safety + validation at startup, not at
runtime. If OPENAI_API_KEY is missing, you get a clear error immediately
rather than a confusing AuthenticationError deep in a benchmark run.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field("sk-placeholder", env="OPENAI_API_KEY")
    openai_org_id: str | None = Field(None, env="OPENAI_ORG_ID")
    primary_model: str = Field("gpt-4o", env="MEMORYBENCH_PRIMARY_MODEL")
    judge_model: str = Field("gpt-4o-mini", env="MEMORYBENCH_JUDGE_MODEL")
    embed_model: str = Field(
        "text-embedding-3-small", env="MEMORYBENCH_EMBED_MODEL"
    )

    # Storage
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    results_dir: str = Field(
        "./experiments/results", env="MEMORYBENCH_RESULTS_DIR"
    )

    # Token budgets
    max_tokens_per_turn: int = Field(
        8000, env="MEMORYBENCH_MAX_TOKENS_PER_TURN"
    )
    daily_spend_limit_usd: float = Field(
        5.0, env="MEMORYBENCH_DAILY_SPEND_LIMIT_USD"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience singleton — reads from .env if present, uses defaults otherwise
settings = get_settings()