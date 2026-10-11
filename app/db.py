"""SQLite persistence for bulk jobs (standard library only)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    job_description TEXT NOT NULL,
    total INTEGER NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS candidates (
    job_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    cid TEXT NOT NULL,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    score REAL,
    matched_skills TEXT,
    PRIMARY KEY (job_id, idx)
);
"""


def sqlite_path(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("DATABASE_URL must look like sqlite:///./resumerank.db")
    return database_url[len(prefix):]


class Database:
    def __init__(self, database_url: str) -> None:
        self.path = sqlite_path(database_url)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_job(self, job_description: str, candidates: list[dict]) -> str:
        job_id = uuid.uuid4().hex
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, job_description, total, created_at) VALUES (?,?,?,?,?)",
                (job_id, "queued", job_description, len(candidates), time.time()),
            )
            conn.executemany(
                "INSERT INTO candidates (job_id, idx, cid, name, text) VALUES (?,?,?,?,?)",
                [(job_id, i, c["id"], c["name"], c["text"]) for i, c in enumerate(candidates)],
            )
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_candidates(self, job_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT idx, cid, name, text FROM candidates WHERE job_id = ? ORDER BY idx", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        finished = time.time() if status in {"done", "failed"} else None
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = COALESCE(?, finished_at) WHERE id = ?",
                (status, error, finished, job_id),
            )

    def save_scores(self, job_id: str, rows: list[tuple[int, float, list[str]]]) -> None:
        """rows: (idx, score, matched_skills). Also advances the progress counter."""
        with self._conn() as conn:
            conn.executemany(
                "UPDATE candidates SET score = ?, matched_skills = ? WHERE job_id = ? AND idx = ?",
                [(score, json.dumps(skills), job_id, idx) for idx, score, skills in rows],
            )
            conn.execute("UPDATE jobs SET done = done + ? WHERE id = ?", (len(rows), job_id))

    def get_results(self, job_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT cid, name, score, matched_skills FROM candidates "
                "WHERE job_id = ? AND score IS NOT NULL ORDER BY score DESC, idx ASC",
                (job_id,),
            ).fetchall()
        return [
            {
                "rank": i + 1,
                "id": r["cid"],
                "name": r["name"],
                "score": r["score"],
                "matched_skills": json.loads(r["matched_skills"] or "[]"),
            }
            for i, r in enumerate(rows)
        ]
