# CPU inference server. GPU variant lives in Dockerfile.gpu (Phase 6).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only torch wheels to keep the image small.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3" "torchvision>=0.18" \
 && pip install --no-cache-dir ".[serve]"

EXPOSE 8000

CMD ["uvicorn", "nutrilens_ml.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
