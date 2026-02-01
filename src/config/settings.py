# src/config/settings.py
from __future__ import annotations

from typing import Optional, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ----------------------------
    # LLM Provider Selection
    # ----------------------------
    llm_provider: Literal["groq", "anthropic", "openai", "gemini", "local"] = "groq"

    # ----------------------------
    # API Keys (OPTIONAL)
    # ----------------------------
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # ----------------------------
    # Model Configuration
    # ----------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "claude-sonnet-4-20250514"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # ----------------------------
    # Retrieval Configuration
    # ----------------------------
    vector_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # ----------------------------
    # Chunking Configuration
    # ----------------------------
    max_chunk_tokens: int = 512
    chunk_overlap: int = 50

    # ----------------------------
    # LLM Runtime Configuration
    # ----------------------------
    max_context_tokens: int = 8000
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2000

    # ----------------------------
    # Data Paths
    # ----------------------------
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    index_dir: str = "data/indices"

    # ----------------------------
    # Validation
    # ----------------------------
    def validate_llm(self) -> None:
        provider_to_key = {
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "local": "LOCAL_OK",
        }

        if provider_to_key[self.llm_provider] in (None, ""):
            raise ValueError(
                f"Missing API key for LLM provider: {self.llm_provider}"
            )

    class Config:
        env_file = ".env"
        extra = "forbid"
