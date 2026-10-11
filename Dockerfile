FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-ml.txt
# Download the model at build time so the first request is not slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:////data/resumerank.db
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
