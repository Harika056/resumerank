"""Ranking: semantic similarity plus a simple, explainable skill-overlap check."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from .cache import EmbeddingCache

# Skills worth surfacing to a recruiter. Matching is whole-word and case-insensitive.
SKILLS = sorted(
    {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "nosql",
        "spring boot", "spring", "fastapi", "flask", "django", "react", "node", "kafka", "redis",
        "postgresql", "mysql", "mongodb", "docker", "kubernetes", "aws", "azure", "gcp",
        "terraform", "jenkins", "git", "ci/cd", "linux", "rest", "graphql", "microservices",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "spark", "airflow",
        "machine learning", "deep learning", "nlp", "transformers", "llm", "rag", "embeddings",
        "pytest", "junit", "selenium", "postman", "jmeter", "prometheus", "grafana", "etl",
        "data structures", "algorithms", "agile", "scrum",
    },
    key=len,
    reverse=True,
)
_SKILL_RES = {s: re.compile(r"(?<![a-z0-9+#])" + re.escape(s) + r"(?![a-z0-9+#])") for s in SKILLS}


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    return [s for s, rx in _SKILL_RES.items() if rx.search(lowered)]


def matched_skills(job_description: str, resume: str) -> list[str]:
    wanted = set(extract_skills(job_description))
    return sorted(s for s in extract_skills(resume) if s in wanted)


def score_candidates(cache: EmbeddingCache, job_description: str, texts: Sequence[str]) -> list[float]:
    """Cosine similarity of every text to the job description (vectors are L2-normalised,
    so the dot product is the cosine). Returned in input order, rounded to 4 places."""
    if not texts:
        return []
    jd_vec = cache.embed([job_description])[0]
    vectors = cache.embed(list(texts))
    sims = vectors @ jd_vec
    return [round(float(np.clip(s, -1.0, 1.0)), 4) for s in sims]
