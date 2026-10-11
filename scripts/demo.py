"""Rank the sample resumes against each sample job description.

Start the API first (see README), then:  python scripts/demo.py [base_url]
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
DATA = Path(__file__).resolve().parent.parent / "data"


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


candidates = [{"name": p.stem, "text": p.read_text()} for p in sorted((DATA / "resumes").glob("*.txt"))]
for jd_file in sorted((DATA / "job_descriptions").glob("*.txt")):
    out = post("/rank", {"job_description": jd_file.read_text(), "candidates": candidates})
    print(f"\n=== {jd_file.stem}  ({out['elapsed_ms']} ms) ===")
    for r in out["results"][:5]:
        print(f"{r['rank']:>2}. {r['name']:<16} score={r['score']:.3f}  skills={', '.join(r['matched_skills'])}")
