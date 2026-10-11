"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Hard upper bound on any single text field (resume or job description).
MAX_TEXT_CHARS = 50_000


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class Settings:
    # Embedding backend: "sentence-transformers" (real) or "hash" (tests/demo only).
    embedder: str = field(default_factory=lambda: os.getenv("EMBEDDER", "sentence-transformers"))
    st_model: str = field(default_factory=lambda: os.getenv("ST_MODEL", "all-MiniLM-L6-v2"))
    # MiniLM truncates at ~256 word pieces; longer texts are split into chunks of
    # this many words and mean-pooled. 0 disables chunking.
    st_chunk_words: int = field(default_factory=lambda: _int("ST_CHUNK_WORDS", 180))
    hash_dim: int = field(default_factory=lambda: _int("HASH_DIM", 384))

    # "memory://" -> in-process cache + in-process queue (single process only).
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache_ttl_seconds: int = field(default_factory=lambda: _int("CACHE_TTL_SECONDS", 0))  # 0 = no expiry
    queue_name: str = field(default_factory=lambda: os.getenv("QUEUE_NAME", "resumerank:bulk"))
    queue_poll_seconds: float = field(default_factory=lambda: _float("QUEUE_POLL_SECONDS", 1.0))

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./resumerank.db"))

    # Run the worker pool inside the API process. Set false when running
    # `python -m app.worker` as a separate process/container.
    worker_embedded: bool = field(default_factory=lambda: _bool("WORKER_EMBEDDED", True))
    worker_threads: int = field(default_factory=lambda: _int("WORKER_THREADS", 4))

    max_bulk: int = field(default_factory=lambda: _int("MAX_BULK", 500))
    max_rank_items: int = field(default_factory=lambda: _int("MAX_RANK_ITEMS", 200))
    max_upload_bytes: int = field(default_factory=lambda: _int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
    max_pdf_pages: int = field(default_factory=lambda: _int("MAX_PDF_PAGES", 30))
