"""Central settings, loaded from environment variables / .env (see .env.example)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Vector DB (Qdrant) ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "policy_chunks"
    qdrant_timeout_s: float = 10.0

    # --- Embeddings (local, free) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # must match the model above
    # Bump this whenever the model or chunking changes; stored in every chunk payload
    # so stale vectors can be detected and re-embedded.
    embedding_version: str = "v1"

    # --- Reranker (local, free) ---
    # Use bge-reranker-base while developing; bge-reranker-large for the final benchmark.
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval ---
    default_top_k: int = 5
    candidate_pool: int = 20  # per-retriever candidates fed into RRF
    rrf_k: int = 60  # standard RRF constant
    rerank_candidates: int = 20  # how many RRF results the cross-encoder rescored

    # --- LLM (OpenAI-compatible endpoint: Ollama / Groq / Gemini) ---
    llm_base_url: str = "http://ollama:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it; set a real key for Groq/Gemini
    llm_model: str = "llama3.2:3b"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 1.0
    llm_temperature: float = 0.0

    # --- Paths ---
    data_dir: Path = Path("data")

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def queries_dir(self) -> Path:
        return self.data_dir / "queries"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    # --- Observability ---
    otel_service_name: str = "hybrid-retrieval-benchmark"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()