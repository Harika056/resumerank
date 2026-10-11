"""Embedding cache keyed by a hash of the text.

The same resume is often submitted more than once (re-uploads, the same candidate for
several jobs). Embedding is the expensive step, so identical text is embedded once.
The cache is an optimisation only: if the store fails, we log and embed anyway.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from typing import Sequence

import numpy as np

from .embeddings import Embedder
from .store import Store

log = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, store: Store, embedder: Embedder, ttl_seconds: int = 0) -> None:
        self.store = store
        self.embedder = embedder
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"rr:emb:{self.embedder.name}:{digest}"

    def _load(self, key: str):
        try:
            raw = self.store.get(key)
        except Exception as exc:  # cache failure must never fail a request
            log.warning("cache get failed: %s", exc)
            return None
        if raw is None:
            return None
        vec = np.frombuffer(raw, dtype=np.float32)
        return vec if vec.shape[0] == self.embedder.dim else None  # ignore stale/corrupt entries

    def _save(self, key: str, vec: np.ndarray) -> None:
        try:
            self.store.set(key, vec.astype(np.float32).tobytes(), self.ttl)
        except Exception as exc:
            log.warning("cache set failed: %s", exc)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts, returning an (n, dim) array. Only cache misses reach the model,
        and duplicate texts within one call are embedded once."""
        n = len(texts)
        out = np.zeros((n, self.embedder.dim), dtype=np.float32)
        pending: dict[str, list[int]] = {}  # key -> row indexes still to fill
        hits = 0
        for i, text in enumerate(texts):
            key = self.key(text)
            vec = self._load(key)
            if vec is not None:
                out[i] = vec
                hits += 1
            else:
                pending.setdefault(key, []).append(i)

        if pending:
            keys = list(pending)
            first_texts = [texts[pending[k][0]] for k in keys]
            vectors = self.embedder.embed(first_texts)
            for key, vec in zip(keys, vectors):
                for row in pending[key]:
                    out[row] = vec
                self._save(key, vec)

        misses = n - hits
        with self._lock:
            self.hits += hits
            self.misses += misses
        return out

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
