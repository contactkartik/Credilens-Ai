# CrediLens AI - all-in-one image for single-container hosts
# (Hugging Face Spaces, Fly.io, a single VPS container, etc.)
#
# Contains: FastAPI backend + RoBERTa model + React frontend (nginx)
#           + SQLite for prediction history (DATABASE_URL env override).
#
# Build context MUST be the repository root:
#     docker build -f deploy/all-in-one/Dockerfile -t credilens-allinone .
#
# Container layout mirrors the repo so REPO_ROOT resolution works:
#     /app/backend -> FastAPI app   (working directory)
#     /app/ml      -> inference code
#     /app/models  -> trained model (baked in for portability)

# ---------- Stage 1: Python deps + app code ----------
FROM python:3.11-slim AS backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    DEBUG=False \
    ROBERTA_MODEL_PATH=models/roberta \
    DATABASE_URL=sqlite:////data/credilens.db

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# CPU-only PyTorch first (default PyPI wheel bundles CUDA, ~2 GB+)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# paddlepaddle publishes aarch64 wheels only on Paddle's own index
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install "paddlepaddle>=3.0" \
            --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ ; \
    fi

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY ml /app/ml

# Trained model - only the files needed for inference (~480 MB).
# The 4.2 GB training checkpoint is intentionally excluded.
COPY models/roberta/config.json models/roberta/model.safetensors \
     models/roberta/tokenizer.json models/roberta/tokenizer_config.json \
     models/roberta/training_info.json /app/models/roberta/

# ---------- Stage 2: frontend build ----------
FROM node:22-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---------- Stage 3: runtime (python + nginx) ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    DEBUG=False \
    ROBERTA_MODEL_PATH=models/roberta \
    DATABASE_URL=sqlite:////data/credilens.db

# nginx serves the frontend and proxies /api -> uvicorn on :8000
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY --from=backend /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin
COPY --from=backend /app /app

COPY --from=frontend /app/dist /usr/share/nginx/html
COPY deploy/all-in-one/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/all-in-one/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# HF Spaces persistent storage lives at /data; create it for other hosts too
RUN mkdir -p /data

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
