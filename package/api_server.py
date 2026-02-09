#!/usr/bin/env python3
"""
Webpage Classification HTTP API Server (Production)

Usage:
    python api_server.py
    python api_server.py --port 8000 --workers 2

Endpoints:
    POST /classify        - Classify a single article
    POST /batch_classify  - Classify multiple articles
    GET  /health          - Health check (with model & GPU status)
    GET  /version         - Service version info
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- Configuration via environment variables (for multi-worker mode) ---
MATRIX_PATH = os.environ.get("WC_MATRIX_PATH", str(PROJECT_ROOT / "models" / "hard_negative_train6000.pt"))
MODEL_PATH = os.environ.get("WC_MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen3-VL-Embedding-8B"))
CORS_ORIGINS = os.environ.get("WC_CORS_ORIGINS", "*").split(",")
CLASSIFY_TIMEOUT = int(os.environ.get("WC_CLASSIFY_TIMEOUT", "30"))

# --- Version ---
SERVICE_VERSION = "1.1.0"

# --- Limits ---
MAX_BODY_SIZE = 1 * 1024 * 1024  # 1MB
MAX_TEXT_LENGTH = 50000
MAX_BATCH_SIZE = 64

# --- Structured JSON Logging ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("api_server")

# --- Global state ---
classifier = None
model_loaded = False
model_load_error = None


def load_model(matrix_path: str, model_path: str):
    """Load embedding model and category matrix."""
    from src.llm.embedding_classifier import QwenEmbeddingClassifier

    clf = QwenEmbeddingClassifier(model_path=model_path, use_template=False)
    clf.load_category_matrix(matrix_path)
    return clf


# --- Request/Response Models ---
class ClassifyRequest(BaseModel):
    articleLink: Optional[str] = Field(default="", max_length=2048)
    title: Optional[str] = Field(default="", max_length=MAX_TEXT_LENGTH)
    text: Optional[str] = Field(default="", max_length=MAX_TEXT_LENGTH)


class BatchClassifyRequest(BaseModel):
    items: list[ClassifyRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class CategoryScore(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    top2: list[CategoryScore]
    latency_ms: float


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total_latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None


class VersionResponse(BaseModel):
    version: str
    categories: list[str]


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, model_loaded, model_load_error
    logger.info(f"Starting model load: matrix={MATRIX_PATH}, model={MODEL_PATH}")
    try:
        classifier = load_model(matrix_path=MATRIX_PATH, model_path=MODEL_PATH)
        model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Model load failed: {e}", exc_info=True)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Webpage Classifier API",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware: Request ID + Body Size + Access Log ---
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    # Request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id

    # Body size check
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": f"Request body too large. Max {MAX_BODY_SIZE} bytes.", "request_id": request_id},
        )

    # Process request
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"

    # Access log
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({latency_ms:.1f}ms)",
        extra={"request_id": request_id},
    )

    return response


# --- Endpoints ---
@app.post("/classify", response_model=ClassifyResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def classify(req: ClassifyRequest, request: Request):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    title = req.title or ""
    text = req.text or ""
    article_link = req.articleLink or ""

    if not title and not text:
        raise HTTPException(status_code=400, detail="At least one of 'title' or 'text' is required.")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(classifier.classify, title=title, url=article_link, content=text),
            timeout=CLASSIFY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Classification timed out ({CLASSIFY_TIMEOUT}s).")
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Classification failed.")

    sorted_scores = sorted(result.scores.items(), key=lambda x: -x[1])
    top2 = [CategoryScore(label=label, score=round(score, 4)) for label, score in sorted_scores[:2]]

    return ClassifyResponse(top2=top2, latency_ms=round(result.latency_ms, 2))


@app.post("/batch_classify", response_model=BatchClassifyResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def batch_classify(req: BatchClassifyRequest, request: Request):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    items_dicts = [
        {"title": item.title or "", "text": item.text or "", "articleLink": item.articleLink or ""}
        for item in req.items
    ]

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(classifier.batch_classify, items_dicts),
            timeout=CLASSIFY_TIMEOUT * len(items_dicts),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Batch classification timed out.")
    except Exception as e:
        logger.error(f"Batch classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch classification failed.")

    classify_responses = []
    for result in results:
        sorted_scores = sorted(result.scores.items(), key=lambda x: -x[1])
        top2 = [CategoryScore(label=label, score=round(score, 4)) for label, score in sorted_scores[:2]]
        classify_responses.append(ClassifyResponse(top2=top2, latency_ms=round(result.latency_ms, 2)))

    total_latency = sum(r.latency_ms for r in results)
    return BatchClassifyResponse(results=classify_responses, total_latency_ms=round(total_latency, 2))


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_used_mb = None
    gpu_memory_total_mb = None

    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_used_mb = int(torch.cuda.memory_allocated(0) / 1024 / 1024)
            gpu_memory_total_mb = int(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024)
        except Exception:
            pass

    status = "ok" if model_loaded else "degraded"
    if model_load_error:
        status = "error"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_memory_total_mb=gpu_memory_total_mb,
    )


@app.get("/version", response_model=VersionResponse)
async def version():
    from src.llm.embedding_classifier import LABELS
    return VersionResponse(version=SERVICE_VERSION, categories=LABELS)


# --- Entrypoint ---
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Webpage Classification API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--matrix", help="Category embedding matrix path")
    parser.add_argument("--model", help="Embedding model path")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--cors-origins", default="*", help="Comma-separated CORS origins")
    parser.add_argument("--timeout", type=int, default=30, help="Classification timeout in seconds")
    args = parser.parse_args()

    # Set env vars so multi-worker mode picks them up
    if args.matrix:
        os.environ["WC_MATRIX_PATH"] = args.matrix
    if args.model:
        os.environ["WC_MODEL_PATH"] = args.model
    if args.cors_origins:
        os.environ["WC_CORS_ORIGINS"] = args.cors_origins
    os.environ["WC_CLASSIFY_TIMEOUT"] = str(args.timeout)

    logger.info(f"Starting server on {args.host}:{args.port} (workers={args.workers})")
    logger.info(f"  POST /classify        - Single classification")
    logger.info(f"  POST /batch_classify  - Batch classification (max {MAX_BATCH_SIZE})")
    logger.info(f"  GET  /health          - Health check")
    logger.info(f"  GET  /version         - Version info")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
