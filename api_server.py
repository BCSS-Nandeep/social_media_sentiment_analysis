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
        Additive (non-breaking): cleaned_text, transliterated_text,
        translation_backend, fallback_used, fallback_reason,
        translation_truncated, sentiment_truncated, low_confidence,
        review_recommended
    GET  /health                          -> {"healthy": bool, "device": str|null, ...}

`/health` gained additive keys only (`stages`, `limits`, `intelligence`);
existing consumers that read `healthy` / `device` are unaffected.

The intelligence layer is a SEPARATE endpoint, so /analyze keeps its exact
request schema, response keys and latency:
    POST /analyze/intelligence
        {"texts": [str, ...],
         "policy_pack"?: {...},   # optional caller taxonomy
         "intent_mode"?: "enum"|"free",
         "timeout_s"?: number}
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
from concurrent.futures import as_completed
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config  # loads .env (HF_TOKEN) via load_dotenv()
from src.fallback_queue import BoundedWorkQueue, FallbackQueueFull
from src.ollama_gate import OllamaCircuitOpen, OllamaGateFull, get_ollama_gate
from src.pipeline_fallback import PipelineFallbackError
from src.translation import TranslationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for noisy in ("transformers", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("sentiment_api")

_pipeline = None  # loaded once at startup, reused across requests
_pipeline_fallback = None  # recoverable translation/sentiment failures only
_pipeline_fallback_queue = None  # bounded service-wide FIFO
_analyzer = None  # stage-3 intelligence layer; None when it failed to construct

# Serializes model inference across the threadpool. Requests queue here; the
# wait is logged separately from the compute so a slow response can be
# attributed to load rather than to the model.
_inference_lock = threading.Lock()

# Correlates the "received" and "completed" log lines for one request.
_request_ids = itertools.count(1)


def _fallback_output_is_english(text: str) -> bool:
    """Validate Ollama output with the loaded general and Roman-Indic LID."""
    if _pipeline is None:
        return False
    if not _inference_lock.acquire(
        timeout=min(30.0, config.INFERENCE_LOCK_TIMEOUT_S)
    ):
        logger.warning("Could not acquire inference lock for fallback validation")
        return False
    try:
        if _pipeline.detector.detect(text) != "en":
            return False
        from src.script_detection import is_latin_script

        if is_latin_script(text):
            refined = _pipeline.roman_detector.detect(text)
            if refined not in (None, "en"):
                logger.warning(
                    "Rejected Ollama fallback output classified as Roman %s",
                    refined,
                )
                return False
        return True
    except Exception as exc:
        logger.warning(
            "Fallback English validation failed (%s)", type(exc).__name__
        )
        return False
    finally:
        _inference_lock.release()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup/shutdown for the pipeline (replaces the deprecated on_event hooks)."""
    global _pipeline
    from src.pipeline import SentimentPipeline

    logger.info(
        "Loading sentiment pipeline (IndicTrans2 primary + NLLB fallback + Cardiff RoBERTa)..."
    )
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

    global _pipeline_fallback, _pipeline_fallback_queue
    if config.PIPELINE_FALLBACK_ENABLED:
        try:
            from src.pipeline_fallback import OllamaPipelineFallback

            _pipeline_fallback = OllamaPipelineFallback(
                english_validator=_fallback_output_is_english
            )
            _pipeline_fallback_queue = BoundedWorkQueue(
                workers=config.PIPELINE_FALLBACK_MAX_WORKERS,
                capacity=config.PIPELINE_FALLBACK_QUEUE_CAPACITY,
                thread_name_prefix="pipeline-fallback",
            )
            logger.info(
                "  pipeline fallback          : %s",
                _pipeline_fallback.describe(),
            )
        except Exception as exc:
            _pipeline_fallback = None
            _pipeline_fallback_queue = None
            logger.error("Pipeline fallback unavailable: %s", exc)

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
        if _pipeline_fallback_queue is not None:
            _pipeline_fallback_queue.shutdown()
            _pipeline_fallback_queue = None
        if _pipeline_fallback is not None:
            _pipeline_fallback.close()
            _pipeline_fallback = None
        if _analyzer is not None:
            _analyzer.close()
            _analyzer = None
        if _pipeline is not None:
            _pipeline.free()
            _pipeline = None
        logger.info("Sentiment pipeline released.")


app = FastAPI(
    title="Social Media Sentiment Analysis", version="1.1.0", lifespan=lifespan
)


class PolicyCategoryModel(BaseModel):
    id: str
    definition: str = ""
    severity: str = ""
    keywords: list[str] = Field(default_factory=list)


class PolicyPackModel(BaseModel):
    name: str = "caller"
    version: str = ""
    fingerprint: str = ""
    unknown_label: Optional[str] = None
    categories: list[PolicyCategoryModel]


class MatchedKeywordModel(BaseModel):
    keyword: str
    weight: int


class AnalyzeRequest(BaseModel):
    texts: list[str]
    # Optional intelligence knobs — ignored by /analyze; used by /analyze/intelligence.
    # Omitting every one reproduces the pre-policy-pack behaviour exactly.
    policy_pack: Optional[PolicyPackModel] = None
    matched_keywords: Optional[list[list[MatchedKeywordModel]]] = None
    intent_mode: Literal["enum", "free"] = "enum"
    timeout_s: Optional[float] = None


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


def _resolve_intelligence_options(req: AnalyzeRequest):
    """Parse optional policy_pack / timeout; raise HTTPException on bad input."""
    from src.intelligence import parse_policy_pack

    pack_dict: dict[str, Any] | None = None
    if req.policy_pack is not None:
        pack_dict = req.policy_pack.model_dump()
    try:
        pack = parse_policy_pack(pack_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    timeout_s = req.timeout_s
    if timeout_s is not None:
        if timeout_s < 5 or timeout_s > 600:
            logger.info("Clamping timeout_s=%s into [5, 600]", timeout_s)
        timeout_s = max(5.0, min(600.0, float(timeout_s)))

    return pack, req.intent_mode, timeout_s


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
        "ollama_gate": get_ollama_gate().stats(),
        "pipeline_fallback": (
            {
                "available": True,
                **_pipeline_fallback.describe(),
                "reachable": _pipeline_fallback.health(),
                "queue": (
                    _pipeline_fallback_queue.stats()
                    if _pipeline_fallback_queue is not None
                    else None
                ),
            }
            if _pipeline_fallback is not None
            else {
                "available": False,
                "enabled": config.PIPELINE_FALLBACK_ENABLED,
            }
        ),
    }


def _run_pipeline(texts: list[str], request_id: int) -> list:
    """Deterministic stage under the inference lock. Shared by both endpoints so
    they cannot drift apart — /analyze/intelligence runs the identical pipeline.

    Lock acquisition has a queue timeout. Individual model-generation calls
    carry the process watchdog so large healthy batches are not given one
    shared wall-clock deadline."""
    queued_at = time.perf_counter()
    if not _inference_lock.acquire(timeout=config.INFERENCE_LOCK_TIMEOUT_S):
        waited_s = time.perf_counter() - queued_at
        logger.error(
            "req %d: gave up after %.0fs waiting for the inference lock — a prior "
            "request appears stuck; failing this request instead of hanging forever",
            request_id, waited_s,
        )
        raise TimeoutError(f"inference lock unavailable after {waited_s:.0f}s")

    try:
        waited_ms = (time.perf_counter() - queued_at) * 1000.0
        if waited_ms > 1000.0:
            logger.info(
                "req %d: waited %.1fs for the inference lock (service is saturated; "
                "run more replicas or batch more posts per request)",
                request_id, waited_ms / 1000.0,
            )
        started = time.perf_counter()
        try:
            results = _pipeline.predict_batch_outcomes(
                texts,
                batch_size=config.BATCH_SIZE,
                translation_batch_size=config.TRANSLATION_BATCH_SIZE,
                request_id=request_id,
            )
        except Exception:
            logger.exception(
                "req %d: failed after %.1fs", request_id, time.perf_counter() - started
            )
            raise
        compute_ms = (time.perf_counter() - started) * 1000.0
    finally:
        _inference_lock.release()

    logger.info(
        "req %d: deterministic pipeline completed %d text(s) in %.0f ms "
        "(%.0f ms/post, %.0f ms queued)",
        request_id, len(results), compute_ms, compute_ms / len(results), waited_ms,
    )
    return results


def _resolve_pipeline_outcomes(outcomes: list, request_id: int) -> list:
    """Resolve only deterministic failures, after the GPU lock is released."""
    from src.pipeline import PipelineFailure

    failures = [
        outcome for outcome in outcomes if isinstance(outcome, PipelineFailure)
    ]
    if not failures:
        return outcomes
    if _pipeline_fallback is None:
        raise PipelineFallbackError(
            f"Deterministic pipeline failed for {len(failures)} post(s) and "
            "Ollama fallback is unavailable"
        )

    unique: dict[tuple[str, str, bool], PipelineFailure] = {}
    for failure in failures:
        key = (
            failure.post_text,
            failure.language,
            failure.was_transliterated,
        )
        unique.setdefault(key, failure)

    logger.warning(
        "req %d: resolving %d deterministic failure(s) through Ollama "
        "(%d unique)",
        request_id,
        len(failures),
        len(unique),
    )
    if _pipeline_fallback_queue is None:
        resolved = {
            key: _pipeline_fallback.resolve(failure)
            for key, failure in unique.items()
        }
    else:
        futures = {}
        try:
            keys = list(unique)
            submitted = _pipeline_fallback_queue.submit_many(
                [
                    (
                        _pipeline_fallback.resolve,
                        (unique[key],),
                        {},
                    )
                    for key in keys
                ],
                timeout_s=config.PIPELINE_FALLBACK_QUEUE_TIMEOUT_S,
            )
            futures = dict(zip(submitted, keys))
        except FallbackQueueFull as exc:
            for future in futures:
                future.cancel()
            logger.error(
                "req %d: Ollama fallback queue saturated (%s)",
                request_id,
                _pipeline_fallback_queue.stats(),
            )
            raise PipelineFallbackError(
                "Ollama fallback queue is at capacity"
            ) from exc
        resolved = {}
        try:
            for future in as_completed(futures):
                resolved[futures[future]] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise

    normalized = []
    for outcome in outcomes:
        if not isinstance(outcome, PipelineFailure):
            normalized.append(outcome)
            continue
        key = (
            outcome.post_text,
            outcome.language,
            outcome.was_transliterated,
        )
        base_failure = unique[key]
        result = resolved[key]
        provider_ms = max(
            0.0,
            result.translation_time_ms
            - base_failure.translation_time_ms,
        )
        translation_ms = round(
            outcome.translation_time_ms + provider_ms, 3
        )
        normalized.append(
            replace(
                result,
                post_text=outcome.post_text,
                language=outcome.language,
                was_transliterated=outcome.was_transliterated,
                translation_time_ms=translation_ms,
                total_time_ms=round(
                    translation_ms + result.sentiment_time_ms, 3
                ),
            )
        )
    return normalized


def _run_pipeline_with_fallback(texts: list[str], request_id: int) -> list:
    outcomes = _run_pipeline(texts, request_id)
    return _resolve_pipeline_outcomes(outcomes, request_id)


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
    try:
        results = _run_pipeline_with_fallback(req.texts, request_id)
    except OllamaGateFull as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after_s))},
        ) from e
    except OllamaCircuitOpen as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after_s))},
        ) from e
    except (TimeoutError, TranslationError, PipelineFallbackError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"results": [r.to_dict() for r in results]}


@app.post("/analyze/intelligence")
def analyze_intelligence(req: AnalyzeRequest):
    """Pipeline output plus the stage-3 assessment, in one record per post.

    The deterministic result is identical to /analyze — same models, same code
    path — with an added `intelligence` object. Intelligence runs *outside* the
    inference lock: it is network-bound work against a separate host, so holding
    the lock through it would block sentiment inference for no reason.

    Optional ``policy_pack`` / ``intent_mode`` / ``timeout_s`` customise the
    Ollama taxonomy and timeout. Omitting them preserves legacy behaviour.
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

    pack, intent_mode, timeout_s = _resolve_intelligence_options(req)

    total_chars = _validate(req.texts)
    request_id = next(_request_ids)
    logger.info(
        "req %d: received %d text(s), %d chars (with intelligence; pack=%s fp=%s mode=%s)",
        request_id, len(req.texts), total_chars,
        pack.name, pack.fingerprint[:19], intent_mode,
    )

    try:
        results = _run_pipeline_with_fallback(req.texts, request_id)
    except OllamaGateFull as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after_s))},
        ) from e
    except OllamaCircuitOpen as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after_s))},
        ) from e
    except (TimeoutError, TranslationError, PipelineFallbackError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    payloads = [r.to_dict() for r in results]

    started = time.perf_counter()
    try:
        records = _analyzer.analyze_batch(
            payloads,
            pack=pack,
            intent_mode=intent_mode,
            timeout_s=timeout_s,
            matched_keywords=[
                [kw.model_dump() for kw in (mks or [])]
                for mks in (req.matched_keywords or [])
            ] if req.matched_keywords else None,
        )
    except OllamaGateFull as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(int(exc.retry_after_s))},
        ) from exc
    except OllamaCircuitOpen as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(int(exc.retry_after_s))},
        ) from exc
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
