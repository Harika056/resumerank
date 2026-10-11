# ResumeRank

A small FastAPI service that ranks resumes against a job description using text embeddings, and explains each ranking with the skills the resume shares with the job.

**Stack:** Python 3.11, FastAPI, sentence-transformers (all-MiniLM-L6-v2), NumPy, Redis, SQLite, pypdf, pytest, Docker, GitHub Actions

## What it does
- `POST /rank` ranks up to 200 resumes against one job description and returns a score and matched skills for each.
- `POST /rank/upload` does the same for uploaded `.txt` and `.pdf` files.
- `POST /jobs` accepts up to 500 resumes, returns `202` straight away, and scores them in background workers. `GET /jobs/{id}` shows progress and, when finished, the ranked results.
- Embeddings are cached in Redis by a hash of the text, so the same resume is embedded once.

## How it works
```
client --> FastAPI --+--> EmbeddingCache --> Redis (vectors)
                     |         |
                     |         +--> embedder (MiniLM, or hash for tests)
                     +--> SQLite (jobs, candidates, scores)
                     +--> Redis queue --> worker threads --> same cache + scoring
```
- **Score** = cosine similarity between the job description vector and the resume vector (vectors are normalised, so it is a dot product).
- **Long resumes:** MiniLM silently truncates around 256 word pieces, so long texts are split into 180-word chunks and the chunk vectors are averaged.
- **Cache key** = embedder name + SHA-256 of the text, so switching models never serves stale vectors. Duplicate texts inside one request are embedded once.
- **If Redis fails**, the cache is skipped and requests still work. Bulk jobs need the queue, so `POST /jobs` returns `503` when it is down.
- **Matched skills** come from a fixed list of about 70 technical skills, matched as whole words.

## Quick start (no model download)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows. On Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
Run the API with the lightweight test embedder and no Redis:
```bash
# Windows PowerShell
$env:EMBEDDER="hash"; $env:REDIS_URL="memory://"
uvicorn app.main:app --port 8000
```
Open http://localhost:8000/docs, or run `python scripts/demo.py` in a second terminal.

> The `hash` embedder only compares words, not meaning. It exists so the project runs and tests offline. Use the real model for any real ranking.

## Run with the real model
```bash
pip install -r requirements-ml.txt      # downloads PyTorch, several hundred MB
$env:EMBEDDER="sentence-transformers"
$env:REDIS_URL="memory://"              # in-process cache and queue, no Redis needed
uvicorn app.main:app --port 8000
```
For a real Redis cache and queue, set `REDIS_URL=redis://localhost:6379/0` instead (or use Docker below).
Or with Docker (builds the image, downloads the model once):
```bash
docker compose up --build
```

## API examples
```bash
curl -X POST http://localhost:8000/rank -H "Content-Type: application/json" -d '{
  "job_description": "Python engineer with FastAPI, Docker and pytest",
  "candidates": [
    {"name": "Asha", "text": "Built FastAPI services in Docker with pytest tests"},
    {"name": "Ben",  "text": "React and CSS front end developer"}
  ]}'

curl -X POST http://localhost:8000/rank/upload -F "job_description=Python engineer" -F "files=@resume1.pdf" -F "files=@resume2.txt"
```

## Tests
```bash
pytest --cov=app --cov-report=term-missing
```
49 tests cover the embedders, cache, stores (in-memory and Redis via fakeredis), database, scoring, workers, and every API route including validation, upload limits, and failure paths. The real sentence-transformers model is not used in tests (a stand-in model checks the chunking code instead).

## Benchmark
With the API running:
```bash
python scripts/benchmark.py http://localhost:8000
```
It measures ranking time with a cold and a warm cache, and bulk-job throughput on 500 generated resumes.
## Results
Measured on my machine (Intel Core i5-10300H, 8 GB RAM).

| Measure | Value |
|---|---|
| Tests passing | 49, 0 failures |
| Line coverage (pytest-cov) | 97% |
| Real-model benchmark | not run yet |


## Data
Everything in `data/` is made up: fictional candidates and job descriptions written for this project. No real resumes are included.

## Limits and honest notes
- Embedding similarity measures how close two texts are in wording and topic. It does not measure ability. Treat the ranking as a way to order a pile for a human to read, not as a hiring decision.
- The skill list is small and fixed. A resume that uses different wording for the same skill will not show it as matched.
- SQLite and in-process workers suit one machine. Several API instances would need a shared database.
