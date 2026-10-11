"""Measure cold vs cached ranking time and bulk-job throughput.

Start the API first (see README), then:  python scripts/benchmark.py [base_url]
It prints plain numbers for YOUR machine. Copy them into the README Results table.
"""
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
DATA = Path(__file__).resolve().parent.parent / "data"
random.seed(7)  # same synthetic resumes every run

SKILLS = ["python", "java", "spring boot", "fastapi", "docker", "kubernetes", "aws", "redis", "kafka",
          "postgresql", "pytorch", "tensorflow", "pandas", "airflow", "spark", "pytest", "junit",
          "selenium", "react", "typescript", "terraform", "jenkins", "transformers", "embeddings"]
ROLES = ["Backend Engineer", "ML Engineer", "QA Engineer", "Data Engineer", "DevOps Engineer", "Full Stack Developer"]


def synthetic_resume(i: int) -> dict:
    skills = random.sample(SKILLS, 6)
    text = (f"{random.choice(ROLES)} with {random.randint(1, 8)} years of experience. "
            f"Worked with {', '.join(skills)}. Built and shipped services, wrote automated tests, "
            f"and collaborated with product teams. Candidate reference number {i}.")
    return {"name": f"candidate-{i:04d}", "text": text}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def bulk(jd, candidates, batch=100):
    t0 = time.perf_counter()
    ids = [call("POST", "/jobs", {"job_description": jd, "candidates": candidates[i:i + batch]})["job_id"]
           for i in range(0, len(candidates), batch)]
    pending = set(ids)
    while pending:
        for jid in list(pending):
            status = call("GET", f"/jobs/{jid}")
            if status["status"] == "failed":
                sys.exit(f"job failed: {status['error']}")
            if status["status"] == "done":
                pending.discard(jid)
        time.sleep(0.2)
    return time.perf_counter() - t0


def main():
    health = call("GET", "/health")
    print(f"embedder: {health['embedder']}   store: {health['store']}")
    jd = (DATA / "job_descriptions" / "ml_engineer.txt").read_text()
    resumes = [synthetic_resume(i) for i in range(500)]

    cold = call("POST", "/rank", {"job_description": jd, "candidates": resumes[:100]})["elapsed_ms"]
    warm = call("POST", "/rank", {"job_description": jd, "candidates": resumes[:100]})["elapsed_ms"]
    print(f"rank 100 resumes, first call (cache cold): {cold:.0f} ms")
    print(f"rank 100 resumes, same call again (cached): {warm:.0f} ms")

    t = bulk(jd, resumes)
    print(f"bulk job, 500 resumes (5 jobs of 100), partly cached: {t:.2f} s = {500 / t:.0f} resumes/s")

    other = [synthetic_resume(i) for i in range(1000, 1500)]
    t = bulk(jd, other)
    print(f"bulk job, 500 NEW resumes (cold cache): {t:.2f} s = {500 / t:.0f} resumes/s")

    stats = call("GET", "/health")["cache"]
    print(f"cache hits: {stats['hits']}, misses: {stats['misses']}, hit rate: {stats['hit_rate']:.0%}")


if __name__ == "__main__":
    main()
