"""api_server.py — HTTP wrapper around the finalized production pipeline
(IndicTrans2 -> Cardiff Twitter RoBERTa), so SOCKEYE's backend can use it as
an alternative sentiment engine (SENTIMENT_ANALYSIS=CUSTOM in backend/.env).

Does not change predict.py / run.py / src/* — this only adds a thin FastAPI
layer around src.pipeline.SentimentPipeline, loaded once at startup, matching
rag_pipeline/api_server.py's config/logging conventions (load_dotenv() +
os.getenv(), same log format, uvicorn launched with an explicit --port).

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8003
"""
from __future__ import annotations

import logging
import sys
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

app = FastAPI(title="Social Media Sentiment Analysis", version="1.0.0")

_pipeline = None  # loaded once at startup, reused across requests


class AnalyzeRequest(BaseModel):
    texts: list[str]


@app.on_event("startup")
def _load_pipeline():
    global _pipeline
    from src.pipeline import SentimentPipeline

    logger.info("Loading sentiment pipeline (IndicTrans2 + Cardiff RoBERTa)...")
    _pipeline = SentimentPipeline(device=config.DEVICE)
    logger.info("Sentiment pipeline ready on %s.", _pipeline.device.type)


@app.get("/health")
def health():
    return {
        "healthy": _pipeline is not None,
        "device": _pipeline.device.type if _pipeline is not None else None,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline still loading")
    if not req.texts:
        return {"results": []}

    results = _pipeline.predict_batch(
        req.texts,
        batch_size=config.BATCH_SIZE,
        translation_batch_size=config.TRANSLATION_BATCH_SIZE,
    )
    return {"results": [r.to_dict() for r in results]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
