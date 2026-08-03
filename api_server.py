"""api_server.py — HTTP wrapper around the finalized production pipeline
(IndicTrans2 -> Cardiff Twitter RoBERTa), so a caller's backend can use it as
an alternative sentiment engine (SENTIMENT_ANALYSIS=CUSTOM in backend/.env).

Does not change predict.py / run.py / src/* model behavior — this only adds a
thin FastAPI layer around src.pipeline.SentimentPipeline, loaded once at
startup, matching rag_pipeline/api_server.py's config/logging conventions
(load_dotenv() + os.getenv(), same log format, uvicorn launched with an
explicit --port).

WIRE CONTRACT — deliberately unchanged, this service is consumed in production:
    POST /analyze  {"texts": [str, ...]}  -> {"results": [ {...}, ... ]}
        result keys: post_text, language, english_text, was_translated,
        was_transliterated, sentiment, confidence, translation_time_ms,
        sentiment_time_ms, total_time_ms
    GET  /health                          -> {"healthy": bool, "device": str|null, ...}

`/health` gained additive keys only (`stages`, `limits`); existing consumers
that read `healthy` / `device` are unaffected.

Concurrency: /analyze is a sync endpoint, so FastAPI dispatches it on the
threadpool and several requests can be in it at once. The pipeline is a single
set of torch modules with per-request mutable state, so inference is serialized
on a module-level lock and the time each request spends queued is logged.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import itertools
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config  # loads .env (HF_TOKEN) via load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("transformers", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("sentiment_api")

_pipeline = None  # loaded once at startup, reused across requests

# Serializes model inference across the threadpool. Requests queue here; the
# wait is logged separately from the compute so a slow response can be
# attributed to load rather than to the model.
_inference_lock = threading.Lock()

# Correlates the "received" and "completed" log lines for one request.
_request_ids = itertools.count(1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown for the pipeline (replaces the deprecated on_event hooks)."""
    global _pipeline
    from src.pipeline import SentimentPipeline

    logger.info("Loading sentiment pipeline (IndicTrans2 + Cardiff RoBERTa)...")
    started = time.perf_counter()
    _pipeline = SentimentPipeline(device=config.DEVICE)
    logger.info(
        "Sentiment pipeline ready on %s in %.1fs.",
        _pipeline.device.type, time.perf_counter() - started,
    )
    # Every optional stage degrades silently when its dependency is missing, so
    # spell out what actually came up rather than just "ready".
    for name, value in _pipeline.stage_status().items():
        logger.info("  stage %-26s : %s", name, value)
    logger.info(
        "  request limits             : %d texts, %d chars/text, %d chars total",
        config.API_MAX_TEXTS, config.API_MAX_TEXT_CHARS, config.API_MAX_TOTAL_CHARS,
    )
    try:
        yield
    finally:
        if _pipeline is not None:
            _pipeline.free()
            _pipeline = None
        logger.info("Sentiment pipeline released.")


app = FastAPI(
    title="Social Media Sentiment Analysis", version="1.0.0", lifespan=lifespan
)


class AnalyzeRequest(BaseModel):
    texts: list[str]


def _validate(texts: list[str]) -> int:
    """Enforce the request-size ceilings; return the total character count.

    Checked here rather than with pydantic validators so the request schema the
    caller sees stays exactly `{"texts": [str]}`, and so the rejection carries
    the offending size and the configured limit. 413 (Payload Too Large) rather
    than 422: the request is well-formed, just too big.
    """
    if len(texts) > config.API_MAX_TEXTS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many texts: {len(texts)} (limit {config.API_MAX_TEXTS}). "
                "Split the batch across several requests."
            ),
        )
    total = 0
    for i, text in enumerate(texts):
        length = len(text)
        if length > config.API_MAX_TEXT_CHARS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"texts[{i}] is {length} characters (limit "
                    f"{config.API_MAX_TEXT_CHARS})."
                ),
            )
        total += length
    if total > config.API_MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Request totals {total} characters (limit "
                f"{config.API_MAX_TOTAL_CHARS})."
            ),
        )
    return total


@app.get("/health")
def health():
    return {
        "healthy": _pipeline is not None,
        "device": _pipeline.device.type if _pipeline is not None else None,
        # Additive: lets a caller confirm the optional stages came up and see
        # the limits it will be held to, without reading the service's logs.
        "stages": _pipeline.stage_status() if _pipeline is not None else None,
        "limits": {
            "max_texts": config.API_MAX_TEXTS,
            "max_text_chars": config.API_MAX_TEXT_CHARS,
            "max_total_chars": config.API_MAX_TOTAL_CHARS,
        },
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    if not req.texts:
        return {"results": []}

    total_chars = _validate(req.texts)
    request_id = next(_request_ids)
    logger.info(
        "req %d: received %d text(s), %d chars", request_id, len(req.texts), total_chars
    )

    queued_at = time.perf_counter()
    with _inference_lock:
        waited_ms = (time.perf_counter() - queued_at) * 1000.0
        if waited_ms > 1000.0:
            logger.info(
                "req %d: waited %.1fs for the inference lock (service is saturated; "
                "run more replicas or batch more posts per request)",
                request_id, waited_ms / 1000.0,
            )
        started = time.perf_counter()
        try:
            results = _pipeline.predict_batch(
                req.texts,
                batch_size=config.BATCH_SIZE,
                translation_batch_size=config.TRANSLATION_BATCH_SIZE,
            )
        except Exception:
            logger.exception(
                "req %d: failed after %.1fs", request_id, time.perf_counter() - started
            )
            raise
        compute_ms = (time.perf_counter() - started) * 1000.0

    logger.info(
        "req %d: completed %d text(s) in %.0f ms (%.0f ms/post, %.0f ms queued)",
        request_id, len(results), compute_ms, compute_ms / len(results), waited_ms,
    )
    return {"results": [r.to_dict() for r in results]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
