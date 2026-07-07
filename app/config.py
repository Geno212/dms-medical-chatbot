"""Central configuration.

Everything is driven by environment variables (optionally loaded from a .env
file) so the same codebase runs against Ollama (default), LM Studio, Groq,
OpenAI, or any other OpenAI-compatible endpoint without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    # --- LLM (any OpenAI-compatible endpoint; defaults target local Ollama) ---
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))

    # --- Embeddings (same endpoint by default; bge-m3 is strong for Arabic) ---
    embed_base_url: str = field(default_factory=lambda: os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")))
    embed_api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", "ollama")))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "bge-m3"))

    # --- Storage ---
    # Backend: "sqlite" (zero-setup default) or "postgres" (Supabase/pgvector).
    # Use APP_DATABASE_URL (preferred) or DATABASE_URL to activate Postgres.
    # APP_DATABASE_URL avoids a collision with Chainlit's own data-layer which
    # intercepts the bare DATABASE_URL env var and exhausts the session pooler.
    database_url: str = field(default_factory=lambda: os.getenv("APP_DATABASE_URL", os.getenv("DATABASE_URL", "")))
    db_backend: str = field(default_factory=lambda: os.getenv(
        "DB_BACKEND", "postgres" if (os.getenv("APP_DATABASE_URL") or os.getenv("DATABASE_URL")) else "sqlite"))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "hospital.db"))))
    dataset_path: Path = field(default_factory=lambda: Path(os.getenv("DATASET_PATH", str(PROJECT_ROOT / "data" / "hospital_dataset.json"))))
    # Conversation state (LangGraph checkpoints) survives restarts here.
    # Set CHECKPOINT_DB=:memory: for ephemeral conversations (tests use this).
    checkpoint_db: str = field(default_factory=lambda: os.getenv(
        "CHECKPOINT_DB", str(PROJECT_ROOT / "data" / "conversations.db")))

    # --- Speech to text ---
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))

    # --- Conversation ---
    history_window: int = field(default_factory=lambda: int(os.getenv("HISTORY_WINDOW", "12")))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "3")))

    # --- Logging (pipeline trace: router decisions, slot extraction, entity
    # resolution, retrieval hits — set LOG_LEVEL=WARNING to silence) ---
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
