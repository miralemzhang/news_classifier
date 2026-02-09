# ============================================================
# Webpage Classification API - Production Dockerfile
# ============================================================
# Build:  docker build -t webpage-classifier .
# Run:    docker run --gpus all -p 8000:8000 webpage-classifier
# ============================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_server.py .
COPY src/ src/

# Models are expected to be mounted at runtime
# docker run -v /path/to/models:/app/models ...
RUN mkdir -p models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python", "api_server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
