"""Pluggable text embedders.

* ``SentenceTransformerEmbedder`` - the real backend (all-MiniLM-L6-v2 by default).
* ``HashEmbedder`` - deterministic feature-hashing bag-of-words. It needs no model
  download, so the whole pipeline is testable offline. It captures lexical overlap
  only, NOT semantic similarity: use it for tests and demos, never for real ranking.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .config import Settings

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#]*")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was
    were will with you your we our they their i me my he she his her not but if than then so
    such can may must should would could also all any each other into over about""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (keeps c++ / c# style suffixes), stopwords removed."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class Embedder(ABC):
    name: str  # part of every cache key, so changing model never serves stale vectors
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""


class HashEmbedder(Embedder):
    BIGRAM_WEIGHT = 0.5

    def __init__(self, dim: int = 384):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.name = f"hash-{dim}-v1"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for bigram in zip(tokens, tokens[1:]):
            key = f"{bigram[0]} {bigram[1]}"
            counts[key] = counts.get(key, 0) + 0  # ensure key exists
        bigram_counts: dict[str, int] = {}
        for a, b in zip(tokens, tokens[1:]):
            key = f"{a} {b}"
            bigram_counts[key] = bigram_counts.get(key, 0) + 1
        for feat, c in counts.items():
            if " " in feat:
                continue
            idx, sign = self._bucket(feat)
            vec[idx] += sign * (1.0 + math.log(c))  # sublinear tf
        for feat, c in bigram_counts.items():
            idx, sign = self._bucket(feat)
            vec[idx] += sign * self.BIGRAM_WEIGHT * (1.0 + math.log(c))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Wraps sentence-transformers. Long texts are chunked and mean-pooled because
    all-MiniLM-L6-v2 silently truncates input at ~256 word pieces."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_words: int = 180, model=None):
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -r requirements-ml.txt` or set EMBEDDER=hash for offline/test use."
                ) from exc
            model = SentenceTransformer(model_name)
        self._model = model
        self._lock = threading.Lock()  # one inference at a time; simple and safe
        self.chunk_words = chunk_words
        self.name = f"sentence-transformers/{model_name}"
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        self.dim = int(dim) if dim else 384

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        if self.chunk_words <= 0 or len(words) <= self.chunk_words:
            return [text]
        return [" ".join(words[i : i + self.chunk_words]) for i in range(0, len(words), self.chunk_words)]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[str] = []
        owners: list[int] = []
        for i, t in enumerate(texts):
            for c in self._chunks(t):
                chunks.append(c)
                owners.append(i)
        with self._lock:
            enc = np.asarray(
                self._model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True),
                dtype=np.float32,
            )
        out = np.zeros((len(texts), enc.shape[1]), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for row, owner in zip(enc, owners):
            out[owner] += row
            counts[owner] += 1
        out /= counts[:, None]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


def get_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder.strip().lower()
    if kind == "hash":
        return HashEmbedder(settings.hash_dim)
    if kind in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(settings.st_model, settings.st_chunk_words)
    raise ValueError(f"Unknown EMBEDDER {settings.embedder!r}; use 'hash' or 'sentence-transformers'")
