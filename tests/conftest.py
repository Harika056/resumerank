from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashEmbedder
from app.main import create_app
from app.store import MemoryStore

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        embedder="hash",
        redis_url="memory://",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        worker_threads=2,
        queue_poll_seconds=0.05,
        max_bulk=50,
        max_rank_items=20,
        max_upload_bytes=20_000,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings, embedder=HashEmbedder(), store=MemoryStore())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jd():
    return (DATA / "job_descriptions" / "ml_engineer.txt").read_text()


@pytest.fixture
def sample_candidates():
    return [
        {"name": p.stem, "text": p.read_text()} for p in sorted((DATA / "resumes").glob("*.txt"))
    ]
