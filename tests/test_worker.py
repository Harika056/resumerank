import threading
import time

from app.cache import EmbeddingCache
from app.db import Database
from app.embeddings import HashEmbedder
from app.store import MemoryStore
from app.worker import WorkerPool


def build(settings):
    store = MemoryStore()
    cache = EmbeddingCache(store, HashEmbedder())
    db = Database(settings.database_url)
    return store, cache, db, WorkerPool(settings, db, cache, store)


def test_process_job_ignores_unknown_job(settings):
    _, _, _, pool = build(settings)
    pool.process_job("missing")  # must not raise


def test_worker_loop_survives_queue_errors_then_processes_job(settings):
    store, cache, db, pool = build(settings)
    calls = {"n": 0}
    real_pop = store.pop

    def flaky_pop(name, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("redis restarting")
        return real_pop(name, timeout)

    store.pop = flaky_pop
    job_id = db.create_job("python", [{"id": "1", "name": "A", "text": "python"}])
    store.push(settings.queue_name, job_id)
    pool.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and db.get_job(job_id)["status"] != "done":
            time.sleep(0.05)
        assert db.get_job(job_id)["status"] == "done"
    finally:
        pool.stop()
    assert not any(t.is_alive() for t in pool._threads)


def test_stop_joins_threads(settings):
    _, _, _, pool = build(settings)
    pool.start()
    assert len(pool._threads) == settings.worker_threads
    pool.stop()
    assert pool._threads == []
