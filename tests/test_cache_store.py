import time

import fakeredis
import numpy as np

from app.cache import EmbeddingCache
from app.embeddings import Embedder, HashEmbedder
from app.store import MemoryStore, RedisStore


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__(32)
        self.embedded = []

    def embed(self, texts):
        self.embedded.extend(texts)
        return super().embed(texts)


def test_second_call_is_served_from_cache():
    emb = CountingEmbedder()
    cache = EmbeddingCache(MemoryStore(), emb)
    first = cache.embed(["alpha beta", "gamma delta"])
    second = cache.embed(["alpha beta", "gamma delta"])
    assert np.array_equal(first, second)
    assert len(emb.embedded) == 2  # model only ran for the first call
    assert cache.stats() == {"hits": 2, "misses": 2, "hit_rate": 0.5}


def test_duplicates_in_one_call_are_embedded_once():
    emb = CountingEmbedder()
    cache = EmbeddingCache(MemoryStore(), emb)
    out = cache.embed(["same text", "other", "same text"])
    assert emb.embedded.count("same text") == 1
    assert np.array_equal(out[0], out[2])


def test_cache_key_includes_embedder_name():
    a = EmbeddingCache(MemoryStore(), HashEmbedder(32))
    b = EmbeddingCache(MemoryStore(), HashEmbedder(64))
    assert a.key("x") != b.key("x")


class BrokenStore(MemoryStore):
    def get(self, key):
        raise ConnectionError("redis down")

    def set(self, key, value, ttl=0):
        raise ConnectionError("redis down")


def test_cache_failure_falls_back_to_embedding():
    emb = CountingEmbedder()
    cache = EmbeddingCache(BrokenStore(), emb)
    out = cache.embed(["still works"])
    assert out.shape == (1, 32)
    assert len(emb.embedded) == 1


def test_wrong_size_cached_vector_is_ignored():
    emb = CountingEmbedder()
    store = MemoryStore()
    cache = EmbeddingCache(store, emb)
    store.set(cache.key("x"), np.zeros(5, dtype=np.float32).tobytes())  # stale/corrupt entry
    out = cache.embed(["x"])
    assert out.shape == (1, 32)
    assert len(emb.embedded) == 1


def test_memory_store_ttl_expires():
    s = MemoryStore()
    s.set("k", b"v", ttl=1)
    assert s.get("k") == b"v"
    time.sleep(1.05)
    assert s.get("k") is None


def test_memory_store_queue_is_fifo_and_pop_times_out():
    s = MemoryStore()
    s.push("q", "a")
    s.push("q", "b")
    assert s.pop("q", 0.05) == "a"
    assert s.pop("q", 0.05) == "b"
    assert s.pop("q", 0.05) is None


def test_redis_store_roundtrip_with_fakeredis():
    store = RedisStore(fakeredis.FakeRedis())
    assert store.ping()
    store.set("k", b"v", ttl=30)
    assert store.get("k") == b"v"
    assert store.get("missing") is None
    store.push("q", "job1")
    assert store.pop("q", 1) == "job1"
    assert store.pop("q", 1) is None


def test_cache_works_on_redis_store():
    emb = CountingEmbedder()
    cache = EmbeddingCache(RedisStore(fakeredis.FakeRedis()), emb, ttl_seconds=60)
    cache.embed(["a b c"])
    cache.embed(["a b c"])
    assert len(emb.embedded) == 1
