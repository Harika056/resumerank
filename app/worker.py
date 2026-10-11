"""Background workers that score bulk jobs.

Run inside the API process (WORKER_EMBEDDED=true, the default) or as a separate process:
    python -m app.worker
"""
from __future__ import annotations

import logging
import threading
import time

from .cache import EmbeddingCache
from .config import Settings
from .db import Database
from .scoring import matched_skills, score_candidates
from .store import Store

log = logging.getLogger(__name__)

BATCH_SIZE = 32


class WorkerPool:
    def __init__(self, settings: Settings, db: Database, cache: EmbeddingCache, store: Store) -> None:
        self.settings = settings
        self.db = db
        self.cache = cache
        self.store = store
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self.settings.worker_threads):
            t = threading.Thread(target=self._loop, name=f"rr-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.store.pop(self.settings.queue_name, self.settings.queue_poll_seconds)
            except Exception as exc:  # e.g. Redis restarting; back off and retry
                log.warning("queue pop failed: %s", exc)
                self._stop.wait(1.0)
                continue
            if job_id:
                self.process_job(job_id)

    def process_job(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            log.warning("job %s not found", job_id)
            return
        try:
            self.db.set_status(job_id, "running")
            candidates = self.db.get_candidates(job_id)
            jd = job["job_description"]
            for start in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[start : start + BATCH_SIZE]
                scores = score_candidates(self.cache, jd, [c["text"] for c in batch])
                rows = [
                    (c["idx"], s, matched_skills(jd, c["text"])) for c, s in zip(batch, scores)
                ]
                self.db.save_scores(job_id, rows)
            self.db.set_status(job_id, "done")
        except Exception as exc:
            log.exception("job %s failed", job_id)
            self.db.set_status(job_id, "failed", error=str(exc)[:500])


def main() -> None:  # pragma: no cover - thin process entry point
    from .embeddings import get_embedder
    from .store import get_store

    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    store = get_store(settings)
    cache = EmbeddingCache(store, get_embedder(settings), settings.cache_ttl_seconds)
    pool = WorkerPool(settings, Database(settings.database_url), cache, store)
    pool.start()
    log.info("worker started with %d threads", settings.worker_threads)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pool.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
