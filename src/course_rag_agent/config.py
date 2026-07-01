from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _path_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "course-rag-agent")

    chroma_dir: Path = _path_env("CHROMA_DIR", "storage/chroma")
    bm25_index_path: Path = _path_env("BM25_INDEX_PATH", "storage/bm25/index.pkl")
    collection_name: str = os.getenv("COLLECTION_NAME", "course_materials")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    allow_hash_embedding_fallback: bool = _bool_env("ALLOW_HASH_EMBEDDING_FALLBACK", True)
    hash_embedding_dim: int = int(os.getenv("HASH_EMBEDDING_DIM", "384"))

    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    top_k: int = int(os.getenv("TOP_K", "5"))
    candidate_k: int = int(os.getenv("CANDIDATE_K", "20"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.65"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.25"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
