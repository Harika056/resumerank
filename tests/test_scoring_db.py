from app.cache import EmbeddingCache
from app.db import Database, sqlite_path
from app.embeddings import HashEmbedder
from app.scoring import extract_skills, matched_skills, score_candidates
from app.store import MemoryStore
import pytest


def test_extract_skills_is_whole_word_and_case_insensitive():
    assert "java" in extract_skills("Built services in Java")
    assert "java" not in extract_skills("Experienced in JavaScript only")
    assert "c++" in extract_skills("Wrote C++ drivers")
    assert "spring boot" in extract_skills("Spring Boot microservices")


def test_matched_skills_is_intersection_with_job():
    jd = "We need Python, Docker and Kafka"
    resume = "Python and Docker and Terraform"
    assert matched_skills(jd, resume) == ["docker", "python"]


def test_score_candidates_orders_relevant_first():
    cache = EmbeddingCache(MemoryStore(), HashEmbedder())
    jd = "python pytorch transformers fastapi model serving"
    scores = score_candidates(cache, jd, ["python pytorch fastapi transformers", "react css html design"])
    assert scores[0] > scores[1]
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_score_candidates_empty():
    cache = EmbeddingCache(MemoryStore(), HashEmbedder())
    assert score_candidates(cache, "jd", []) == []


def test_sqlite_path_validation():
    assert sqlite_path("sqlite:///./x.db") == "./x.db"
    with pytest.raises(ValueError):
        sqlite_path("postgresql://x")


def test_database_job_lifecycle(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'a.db'}")
    job_id = db.create_job("jd text", [{"id": "1", "name": "A", "text": "t1"}, {"id": "2", "name": "B", "text": "t2"}])
    job = db.get_job(job_id)
    assert job["status"] == "queued" and job["total"] == 2 and job["done"] == 0
    assert [c["name"] for c in db.get_candidates(job_id)] == ["A", "B"]

    db.save_scores(job_id, [(0, 0.2, ["python"]), (1, 0.9, [])])
    db.set_status(job_id, "done")
    job = db.get_job(job_id)
    assert job["done"] == 2 and job["finished_at"] is not None
    results = db.get_results(job_id)
    assert [r["name"] for r in results] == ["B", "A"]
    assert results[0]["rank"] == 1
    assert results[1]["matched_skills"] == ["python"]


def test_database_unknown_job(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'b.db'}")
    assert db.get_job("nope") is None
    assert db.get_results("nope") == []
