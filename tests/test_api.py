import io
import time
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

DATA = Path(__file__).resolve().parent.parent / "data"


def make_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, text)
    c.save()
    return buf.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["store"] == "memory"
    assert body["embedder"].startswith("hash-")


def test_rank_orders_ml_resumes_above_unrelated(client, sample_jd, sample_candidates):
    r = client.post("/rank", json={"job_description": sample_jd, "candidates": sample_candidates})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["rank"] for x in results] == list(range(1, len(results) + 1))
    # The test embedder is keyword-based, so only the clear-cut cases are asserted.
    assert {x["name"] for x in results[:3]} == {"asha_menon", "rohan_iyer", "vikram_rao"}
    assert results[-1]["name"] != "asha_menon"
    assert results[0]["score"] >= results[-1]["score"]
    top = next(x for x in results if x["name"] == "asha_menon")
    assert {"pytorch", "fastapi", "docker", "pytest"} <= set(top["matched_skills"])


def test_rank_backend_job_prefers_java_candidates(client, sample_candidates):
    jd = (DATA / "job_descriptions" / "backend_java.txt").read_text()
    results = client.post("/rank", json={"job_description": jd, "candidates": sample_candidates}).json()["results"]
    assert {x["name"] for x in results[:2]} == {"priya_nair", "arjun_das"}


def test_rank_qa_job_prefers_qa_candidates(client, sample_candidates):
    jd = (DATA / "job_descriptions" / "qa_automation.txt").read_text()
    results = client.post("/rank", json={"job_description": jd, "candidates": sample_candidates}).json()["results"]
    assert {x["name"] for x in results[:2]} == {"divya_reddy", "imran_khan"}


def test_second_identical_rank_hits_cache(client, sample_jd, sample_candidates):
    body = {"job_description": sample_jd, "candidates": sample_candidates}
    client.post("/rank", json=body)
    before = client.get("/health").json()["cache"]
    client.post("/rank", json=body)
    after = client.get("/health").json()["cache"]
    assert after["misses"] == before["misses"]
    assert after["hits"] > before["hits"]


@pytest.mark.parametrize(
    "payload",
    [
        {"job_description": "", "candidates": [{"name": "A", "text": "x"}]},
        {"job_description": "jd", "candidates": []},
        {"job_description": "jd", "candidates": [{"name": "A", "text": "   "}]},
        {"job_description": "jd", "candidates": [{"name": "", "text": "x"}]},
        {"job_description": "jd"},
        {"job_description": "x" * 50_001, "candidates": [{"name": "A", "text": "x"}]},
    ],
)
def test_rank_validation_errors(client, payload):
    assert client.post("/rank", json=payload).status_code == 422


def test_rank_rejects_too_many_candidates(client):
    cands = [{"name": f"c{i}", "text": "python"} for i in range(21)]
    r = client.post("/rank", json={"job_description": "python", "candidates": cands})
    assert r.status_code == 422
    assert "too many" in r.json()["detail"]


def test_rank_assigns_default_ids(client):
    r = client.post(
        "/rank",
        json={"job_description": "python", "candidates": [{"name": "A", "text": "python"}, {"id": "x9", "name": "B", "text": "java"}]},
    )
    ids = {x["name"]: x["id"] for x in r.json()["results"]}
    assert ids == {"A": "1", "B": "x9"}


def test_upload_txt_and_pdf(client):
    files = [
        ("files", ("ml_person.txt", b"python pytorch fastapi transformers docker", "text/plain")),
        ("files", ("web_person.pdf", make_pdf("react css html design"), "application/pdf")),
    ]
    r = client.post("/rank/upload", data={"job_description": "python pytorch transformers"}, files=files)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert results[0]["name"] == "ml_person"
    assert {x["name"] for x in results} == {"ml_person", "web_person"}


def test_upload_rejects_unsupported_type(client):
    files = [("files", ("cv.docx", b"whatever", "application/octet-stream"))]
    r = client.post("/rank/upload", data={"job_description": "python"}, files=files)
    assert r.status_code == 415


def test_upload_rejects_oversized_file(client):
    files = [("files", ("big.txt", b"a" * 20_001, "text/plain"))]
    r = client.post("/rank/upload", data={"job_description": "python"}, files=files)
    assert r.status_code == 413


def test_upload_rejects_corrupt_pdf_and_empty_text(client):
    bad = client.post("/rank/upload", data={"job_description": "python"}, files=[("files", ("x.pdf", b"not a pdf", "application/pdf"))])
    assert bad.status_code == 415
    empty = client.post("/rank/upload", data={"job_description": "python"}, files=[("files", ("e.txt", b"   ", "text/plain"))])
    assert empty.status_code == 422


def wait_done(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in {"done", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_bulk_job_runs_in_background_and_returns_ranked_results(client, sample_jd, sample_candidates):
    r = client.post("/jobs", json={"job_description": sample_jd, "candidates": sample_candidates})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    body = wait_done(client, job_id)
    assert body["status"] == "done"
    assert body["done"] == body["total"] == len(sample_candidates)
    assert body["results"][0]["rank"] == 1
    assert {x["name"] for x in body["results"][:3]} == {"asha_menon", "rohan_iyer", "vikram_rao"}
    assert body["seconds"] >= 0


def test_bulk_job_with_more_than_one_batch(client):
    cands = [{"name": f"c{i}", "text": f"python docker item{i}"} for i in range(50)]
    body = wait_done(client, client.post("/jobs", json={"job_description": "python docker", "candidates": cands}).json()["job_id"])
    assert body["done"] == 50 and len(body["results"]) == 50


def test_bulk_job_limit_and_unknown_job(client):
    cands = [{"name": f"c{i}", "text": "python"} for i in range(51)]
    assert client.post("/jobs", json={"job_description": "python", "candidates": cands}).status_code == 422
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_job_failure_is_recorded(client, monkeypatch):
    import app.worker as worker

    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(worker, "score_candidates", boom)
    job_id = client.post("/jobs", json={"job_description": "python", "candidates": [{"name": "A", "text": "python"}]}).json()["job_id"]
    body = wait_done(client, job_id)
    assert body["status"] == "failed"
    assert "model exploded" in body["error"]
    assert body["results"] == []


def test_enqueue_failure_returns_503(settings):
    from fastapi.testclient import TestClient

    from app.embeddings import HashEmbedder
    from app.main import create_app
    from app.store import MemoryStore

    class NoQueue(MemoryStore):
        def push(self, queue_name, item):
            raise ConnectionError("down")

    app = create_app(settings, embedder=HashEmbedder(), store=NoQueue())
    with TestClient(app) as c:
        r = c.post("/jobs", json={"job_description": "python", "candidates": [{"name": "A", "text": "python"}]})
        assert r.status_code == 503
