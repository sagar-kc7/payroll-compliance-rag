# Build-time note: the retrieval index is built INSIDE this image from
# the committed data/processed/chunks.jsonl (the corpus doesn't change
# per-deployment, so baking it in keeps container startup fast — the
# only remaining cold-start cost at runtime is the cross-encoder
# reranker model, handled by the FastAPI lifespan warmup in
# src/api/main.py, same as local dev).

FROM python:3.12-slim

WORKDIR /app

# hnswlib (a chromadb dependency) sometimes needs to compile from
# source if no prebuilt wheel matches this exact platform — included
# defensively since I can't verify the build myself (see PR description).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src/ ./src/
COPY data/processed/chunks.jsonl ./data/processed/chunks.jsonl

# Same install pattern already proven in .github/workflows/eval.yml —
# --system since there's no need for a venv inside a container.
RUN uv pip install --system -e .

ENV PYTHONPATH=/app

RUN python src/retrieval/baseline.py

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]