# ---------- base ----------
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- dev ----------
FROM base as dev

# Add dev tools
RUN pip install --no-cache-dir uvicorn[standard] watchdog

# App code
COPY ./app /app/app

# Create non-root user WITH home (-m creates /home/fastapi)
RUN groupadd -r fastapi && useradd -m -r -g fastapi fastapi

# Point Hugging Face caches to user's home (writable)
ENV HOME=/home/fastapi \
    HF_HOME=/home/fastapi/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/fastapi/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/home/fastapi/.cache/huggingface/sentencetransformers

# Make sure cache dirs exist and are owned correctly
RUN mkdir -p $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME $HF_HOME/hub && \
    chown -R fastapi:fastapi /home/fastapi /app

USER fastapi
EXPOSE 8000

# Dev command (hot reload + access logs)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info", "--access-log"]

# ---------- prod ----------
FROM base as prod

COPY ./app /app/app

# Create non-root user WITH home, then set cache envs
RUN groupadd -r fastapi && useradd -m -r -g fastapi fastapi
ENV HOME=/home/fastapi \
    HF_HOME=/home/fastapi/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/fastapi/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/home/fastapi/.cache/huggingface/sentencetransformers
RUN mkdir -p $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME $HF_HOME/hub && \
    chown -R fastapi:fastapi /home/fastapi /app

# Switch to app user BEFORE pre-downloading so files are owned by it
USER fastapi

# (Optional but recommended) Pre-download your model to avoid runtime locks/races
# Replace the model id below with the one(s) you actually use.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")
print("model cached")
PY

EXPOSE 8000

# Prod command — add access logs so you can see hits
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info", "--access-log"]
CMD ["python", "-m", "app.main_runpod"]