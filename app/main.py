"""ResumeRank HTTP API."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .cache import EmbeddingCache
from .config import MAX_TEXT_CHARS, Settings
from .db import Database
from .embeddings import Embedder, get_embedder
from .files import UnsupportedFile, extract_text
from .scoring import matched_skills, score_candidates
from .store import Store, get_store
from .worker import WorkerPool

log = logging.getLogger(__name__)


class CandidateIn(BaseModel):
    id: Optional[str] = Field(None, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class RankRequest(BaseModel):
    job_description: str = Field(..., max_length=MAX_TEXT_CHARS)
    candidates: list[CandidateIn] = Field(..., min_length=1)

    @field_validator("job_description")
    @classmethod
    def jd_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("job_description must not be blank")
        return v


def _normalise(candidates: list[CandidateIn]) -> list[dict]:
    return [{"id": c.id or str(i + 1), "name": c.name, "text": c.text} for i, c in enumerate(candidates)]


def _rank(cache: EmbeddingCache, job_description: str, candidates: list[dict]) -> list[dict]:
    scores = score_candidates(cache, job_description, [c["text"] for c in candidates])
    rows = [
        {
            "id": c["id"],
            "name": c["name"],
            "score": s,
            "matched_skills": matched_skills(job_description, c["text"]),
        }
        for c, s in zip(candidates, scores)
    ]
    rows.sort(key=lambda r: -r["score"])  # stable: ties keep input order
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def create_app(
    settings: Optional[Settings] = None,
    embedder: Optional[Embedder] = None,
    store: Optional[Store] = None,
) -> FastAPI:
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st = store or get_store(cfg)
        emb = embedder or get_embedder(cfg)
        app.state.settings = cfg
        app.state.store = st
        app.state.cache = EmbeddingCache(st, emb, cfg.cache_ttl_seconds)
        app.state.db = Database(cfg.database_url)
        app.state.pool = None
        if cfg.worker_embedded:
            app.state.pool = WorkerPool(cfg, app.state.db, app.state.cache, st)
            app.state.pool.start()
        yield
        if app.state.pool:
            app.state.pool.stop()

    app = FastAPI(title="ResumeRank", version=__version__, lifespan=lifespan)

    def _check_count(n: int, limit: int, what: str) -> None:
        if n > limit:
            raise HTTPException(status_code=422, detail=f"too many {what}: {n} (max {limit})")

    @app.get("/health")
    def health(request: Request):
        s = request.app.state
        return {
            "status": "ok",
            "version": __version__,
            "embedder": s.cache.embedder.name,
            "store": s.store.backend,
            "store_reachable": s.store.ping(),
            "cache": s.cache.stats(),
        }

    @app.post("/rank")
    def rank(body: RankRequest, request: Request):
        s = request.app.state
        _check_count(len(body.candidates), s.settings.max_rank_items, "candidates")
        t0 = time.perf_counter()
        results = _rank(s.cache, body.job_description, _normalise(body.candidates))
        return {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 2), "results": results}

    @app.post("/rank/upload")
    def rank_upload(
        request: Request,
        job_description: str = Form(...),
        files: list[UploadFile] = File(...),
    ):
        s = request.app.state
        if not job_description.strip() or len(job_description) > MAX_TEXT_CHARS:
            raise HTTPException(422, "job_description must be 1-%d characters" % MAX_TEXT_CHARS)
        _check_count(len(files), s.settings.max_rank_items, "files")
        candidates = []
        for i, f in enumerate(files):
            data = f.file.read(s.settings.max_upload_bytes + 1)
            if len(data) > s.settings.max_upload_bytes:
                raise HTTPException(413, f"{f.filename}: larger than {s.settings.max_upload_bytes} bytes")
            try:
                text = extract_text(f.filename or "", data, s.settings.max_pdf_pages)
            except UnsupportedFile as exc:
                raise HTTPException(415, str(exc))
            if not text:
                raise HTTPException(422, f"{f.filename}: no text found")
            name = (f.filename or f"file-{i + 1}").rsplit(".", 1)[0]
            candidates.append({"id": str(i + 1), "name": name, "text": text[:MAX_TEXT_CHARS]})
        t0 = time.perf_counter()
        results = _rank(s.cache, job_description, candidates)
        return {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 2), "results": results}

    @app.post("/jobs", status_code=202)
    def create_job(body: RankRequest, request: Request):
        s = request.app.state
        _check_count(len(body.candidates), s.settings.max_bulk, "candidates")
        job_id = s.db.create_job(body.job_description, _normalise(body.candidates))
        try:
            s.store.push(s.settings.queue_name, job_id)
        except Exception as exc:
            s.db.set_status(job_id, "failed", error="queue unavailable")
            log.error("could not enqueue job %s: %s", job_id, exc)
            raise HTTPException(503, "queue unavailable, try again shortly")
        return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        s = request.app.state
        job = s.db.get_job(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        body = {
            "job_id": job["id"],
            "status": job["status"],
            "total": job["total"],
            "done": job["done"],
            "error": job["error"],
        }
        if job["finished_at"]:
            body["seconds"] = round(job["finished_at"] - job["created_at"], 3)
        body["results"] = s.db.get_results(job_id) if job["status"] == "done" else []
        return body

    return app


app = create_app()
