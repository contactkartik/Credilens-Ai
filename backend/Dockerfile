# CrediLens AI - backend image (FastAPI + RoBERTa + OCR)
#
# Build context MUST be the repository root (the backend imports the `ml/`
# package, and the trained model lives in `models/`):
#     docker build -f backend/Dockerfile -t credilens-backend .
#
# Container layout (mirrors the repo, so REPO_ROOT resolution in
# credibility_service.py works unchanged):
#     /app/backend   -> FastAPI app   (working directory)
#     /app/ml        -> inference code
#     /app/models    -> trained model (mounted as a volume, NOT baked in)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    DEBUG=False \
    ROBERTA_MODEL_PATH=models/roberta

# System libraries needed by PaddleOCR / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# CPU-only PyTorch first (the default PyPI wheel bundles CUDA and is ~2 GB+).
# requirements.txt then sees torch as already satisfied.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# paddlepaddle publishes aarch64 (ARM64, e.g. Oracle Cloud Ampere) wheels only
# on Paddle's own index, not PyPI. Install it first on ARM builds so
# requirements.txt sees it as already satisfied; x86_64 uses PyPI as usual.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install "paddlepaddle>=3.0" \
            --index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ ; \
    fi

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Application code (models/ is mounted at runtime - see docker-compose.yml)
COPY backend/app ./app
COPY ml /app/ml

EXPOSE 8000

# The model is loaded at startup (can take 1-2 minutes on first start)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
