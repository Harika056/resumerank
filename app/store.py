"""Key/value + queue storage behind one small interface.

* ``MemoryStore``  - in-process, for tests and single-process demos (REDIS_URL=memory://).
* ``RedisStore``   - real Redis, used for the embedding cache and the bulk-job queue.
"""
from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from .config import Settings


class Store(ABC):
    backend: str

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]: ...

    @abstractmethod
    def set(self, key: str, value: bytes, ttl: int = 0) -> None: ...

    @abstractmethod
    def push(self, queue_name: str, item: str) -> None: ...

    @abstractmethod
    def pop(self, queue_name: str, timeout: float) -> Optional[str]: ...

    @abstractmethod
    def ping(self) -> bool: ...


class MemoryStore(Store):
    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float]] = {}  # key -> (value, expires_at or 0)
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires = entry
            if expires and expires < time.monotonic():
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: bytes, ttl: int = 0) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl if ttl > 0 else 0.0)

    def _queue(self, name: str) -> queue.Queue:
        with self._lock:
            return self._queues.setdefault(name, queue.Queue())

    def push(self, queue_name: str, item: str) -> None:
        self._queue(queue_name).put(item)

    def pop(self, queue_name: str, timeout: float) -> Optional[str]:
        try:
            return self._queue(queue_name).get(timeout=max(timeout, 0.01))
        except queue.Empty:
            return None

    def ping(self) -> bool:
        return True


class RedisStore(Store):
    backend = "redis"

    def __init__(self, client) -> None:
        self._r = client

    @classmethod
    def from_url(cls, url: str) -> "RedisStore":
        import redis

        return cls(redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=5))

    def get(self, key: str) -> Optional[bytes]:
        return self._r.get(key)

    def set(self, key: str, value: bytes, ttl: int = 0) -> None:
        if ttl > 0:
            self._r.set(key, value, ex=ttl)
        else:
            self._r.set(key, value)

    def push(self, queue_name: str, item: str) -> None:
        self._r.lpush(queue_name, item)

    def pop(self, queue_name: str, timeout: float) -> Optional[str]:
        # BRPOP takes whole seconds; round up so a short poll interval still blocks briefly.
        result = self._r.brpop(queue_name, timeout=max(1, int(round(timeout))))
        if not result:
            return None
        raw = result[1]
        return raw.decode() if isinstance(raw, bytes) else raw

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


def get_store(settings: Settings) -> Store:
    if settings.redis_url.startswith("memory://"):
        return MemoryStore()
    return RedisStore.from_url(settings.redis_url)
