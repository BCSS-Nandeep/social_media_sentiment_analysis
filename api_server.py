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

`/health` gained additive keys only (`stages`, `limits`, `intelligence`);
existing consumers that read `healthy` / `device` are unaffected.

The intelligence layer is a SEPARATE endpoint, so /analyze keeps its exact
request schema, response keys and latency:
    POST /analyze/intelligence  {"texts": [str, ...]}
        -> {"results": [ {<every /analyze key>, "intelligence": {...}} ]}

That endpoint runs the identical deterministic pipeline and then adds the
second-stage assessment. Sentiment is never recomputed or overridden by it, and
if the intelligence provider is unreachable each record still carries a
well-formed "insufficient evidence" assessment alongside intact sentiment.

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
_analyzer = None  # stage-3 intelligence layer; None when it failed to construct

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

    # Stage 3 is optional and must never be able to take down /analyze, so it is
    # imported and constructed defensively — a missing dependency or a bad
    # OLLAMA_BASE_URL degrades to "intelligence unavailable", not a failed boot.
    global _analyzer
    try:
        from src.intelligence import IntelligenceAnalyzer

        _analyzer = IntelligenceAnalyzer()
        reachable = _analyzer.health()
        logger.info(
            "  intelligence               : %s (%s at startup)",
            _analyzer.describe(),
            "reachable" if reachable else "NOT reachable — will retry per request",
        )
    except Exception as exc:
        _analyzer = None
        logger.error(
            "Intelligence layer unavailable (%s) — /analyze is unaffected; "
            "/analyze/intelligence will return 503.", exc,
        )

    try:
        yield
    finally:
        if _analyzer is not None:
            _analyzer.close()
            _analyzer = None
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
        "intelligence": (
            {"available": True, **_analyzer.describe(), "reachable": _analyzer.health()}
            if _analyzer is not None
            else {"available": False}
        ),
    }


def _run_pipeline(texts: list[str], request_id: int) -> list:
    """Deterministic stage under the inference lock. Shared by both endpoints so
    they cannot drift apart — /analyze/intelligence runs the identical pipeline."""
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
                texts,
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
        "req %d: pipeline completed %d text(s) in %.0f ms (%.0f ms/post, %.0f ms queued)",
        request_id, len(results), compute_ms, compute_ms / len(results), waited_ms,
    )
    return results


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
    results = _run_pipeline(req.texts, request_id)
    return {"results": [r.to_dict() for r in results]}


@app.post("/analyze/intelligence")
def analyze_intelligence(req: AnalyzeRequest):
    """Pipeline output plus the stage-3 assessment, in one record per post.

    The deterministic result is identical to /analyze — same models, same code
    path — with an added `intelligence` object. Intelligence runs *outside* the
    inference lock: it is network-bound work against a separate host, so holding
    the lock through it would block sentiment inference for no reason.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    if _analyzer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Intelligence layer unavailable — check OLLAMA_BASE_URL and the "
                "service logs. /analyze is unaffected."
            ),
        )
    if not req.texts:
        return {"results": []}

    total_chars = _validate(req.texts)
    request_id = next(_request_ids)
    logger.info(
        "req %d: received %d text(s), %d chars (with intelligence)",
        request_id, len(req.texts), total_chars,
    )

    results = _run_pipeline(req.texts, request_id)
    payloads = [r.to_dict() for r in results]

    started = time.perf_counter()
    records = _analyzer.analyze_batch(payloads)
    logger.info(
        "req %d: intelligence completed in %.0f ms",
        request_id, (time.perf_counter() - started) * 1000.0,
    )

    return {
        "results": [
            {**payload, "intelligence": record.to_dict()}
            for payload, record in zip(payloads, records)
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
