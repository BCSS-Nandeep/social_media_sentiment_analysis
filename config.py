"""Central configuration for the two-stage translate-then-classify benchmark.

Stage 1 — Translation (Indian language -> English), two competing pipelines.
Stage 2 — English sentiment classification, two competing models.

NOTE ON MODEL CHOICE
--------------------
IndicBERT v2 and MuRIL are *encoder-only* masked-LM models: they have no
decoder and cannot translate. The translation stage therefore uses real
Indic->English translation models:
  * AI4Bharat pipeline : IndicTrans2 (distilled, 200M) — AI4Bharat's actual
    translation model.
  * Second pipeline    : NLLB-200 (distilled, 600M). Swap ``hf_id`` below for
    ``google/madlad400-3b-mt`` (family="madlad") if you specifically want
    Google's open translator — it is ~12 GB and much slower.
Every model is configured here; no module hardcodes a model name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Loads .env (e.g. HF_TOKEN for the gated IndicTrans2 model) into the process
# environment — same load_dotenv() + os.getenv() convention as rag_pipeline.
load_dotenv()

# --------------------------------------------------------------------------- #
# Paths — always resolved relative to this file (no hardcoded absolute paths).
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

DEFAULT_DATASET: Path = DATA_DIR / "dataset.csv"

# --------------------------------------------------------------------------- #
# Task definition
# --------------------------------------------------------------------------- #
LABELS: list[str] = ["Negative", "Neutral", "Positive"]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

REQUIRED_COLUMNS: tuple[str, ...] = ("post_text", "ground_truth_sentiment")
OPTIONAL_COLUMNS: tuple[str, ...] = ("id", "platform", "topic", "sentiment_score")

# Optional column with human English references; enables reference-based
# BLEU / chrF / COMET. Without it the pipeline falls back to reference-free
# scoring (COMET-QE when installed) plus cross-pipeline agreement.
REFERENCE_COLUMN: str = "reference_translation"


# --------------------------------------------------------------------------- #
# Stage 1 — translation pipelines
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranslationConfig:
    """Static description of one benchmarked translation pipeline."""

    key: str            # short identifier used in file names
    display_name: str   # human-readable name used in reports
    hf_id: str          # HuggingFace hub id
    family: str         # "indictrans2" | "nllb" | "madlad"


TRANSLATION_MODELS: dict[str, TranslationConfig] = {
    "indictrans2": TranslationConfig(
        key="indictrans2",
        display_name="IndicTrans2 (AI4Bharat)",
        hf_id="ai4bharat/indictrans2-indic-en-dist-200M",
        family="indictrans2",
    ),
    "nllb": TranslationConfig(
        key="nllb",
        display_name="NLLB-200 (600M)",
        hf_id="facebook/nllb-200-distilled-600M",
        family="nllb",
    ),
}

# ISO 639-1 -> FLORES-200 codes used by both translator families.
FLORES_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}
TARGET_FLORES: str = "eng_Latn"

TRANSLATION_BATCH_SIZE: int = int(os.getenv("SENTIMENT_TRANSLATION_BATCH_SIZE", "8"))
TRANSLATION_MAX_LENGTH: int = 256
TRANSLATION_NUM_BEAMS: int = 1  # raise to 4-5 for higher quality (slower on CPU)
# Hugging Face's `max_time` is a cooperative generation bound: generation
# checks it between decoding passes. The process watchdog below remains the hard
# bound for a CUDA/kernel call that never returns to Python.
GENERATION_MAX_TIME_S: float = float(
    os.getenv("SENTIMENT_GENERATION_MAX_TIME_S", "30")
)

# Runaway-generation guard. A translation is never much longer than its source,
# but a degenerate decode (e.g. IndicTrans2 running without IndicTransToolkit's
# preprocessing) can loop until it hits TRANSLATION_MAX_LENGTH — paying the full
# 256-token decode for a 10-word post. Cap each batch at
# ``ratio * longest_source_in_batch + margin`` new tokens instead. Generous by
# default, so it only bites on runaway decodes; every clamp is logged with the
# source length so the effect is measurable. Set the ratio to 0 to disable and
# fall back to a flat TRANSLATION_MAX_LENGTH budget.
TRANSLATION_LENGTH_RATIO: float = float(os.getenv("SENTIMENT_TRANSLATION_LENGTH_RATIO", "2.5"))
TRANSLATION_LENGTH_MARGIN: int = int(os.getenv("SENTIMENT_TRANSLATION_LENGTH_MARGIN", "32"))
# Translation output sanity limits. IndicTrans2 output that remains primarily
# in the source script, expands implausibly, or loops one token is delegated to
# the NLLB fallback instead of being accepted as English.
TRANSLATION_OUTPUT_MAX_RATIO: float = float(
    os.getenv("SENTIMENT_TRANSLATION_OUTPUT_MAX_RATIO", "4.0")
)
TRANSLATION_REPEAT_TOKEN_RATIO: float = float(
    os.getenv("SENTIMENT_TRANSLATION_REPEAT_TOKEN_RATIO", "0.5")
)

# Translation memoization, keyed on (source language, exact source text).
# Translation dominates end-to-end latency, so a caller that retries a post
# after a client-side timeout gets the completed work for free instead of
# paying for it twice. 0 disables the cache.
TRANSLATION_CACHE_SIZE: int = int(os.getenv("SENTIMENT_TRANSLATION_CACHE_SIZE", "2048"))

# Winner rule for the translation stage, evaluated in order. Metrics that are
# unavailable for the current run (e.g. COMET not installed) are skipped.
TRANSLATION_WINNER_PRIORITY: tuple[str, ...] = ("comet", "chrf", "bleu", "avg_time_ms")

# Optional COMET checkpoints (require `pip install unbabel-comet`; CometKiwi
# additionally requires a HuggingFace login as it is a gated model).
COMET_REF_MODEL: str = "Unbabel/wmt22-comet-da"
COMET_QE_MODEL: str = "Unbabel/wmt22-cometkiwi-da"


# --------------------------------------------------------------------------- #
# Stage 2 — English sentiment models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SentimentModelConfig:
    """Static description of one benchmarked sentiment model."""

    key: str
    display_name: str
    hf_id: str
    checkpoint_dir: Path  # local override checkpoint (auto-loaded when present)


SENTIMENT_MODELS: dict[str, SentimentModelConfig] = {
    "cardiff": SentimentModelConfig(
        key="cardiff",
        display_name="Cardiff Twitter RoBERTa",
        hf_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        checkpoint_dir=MODELS_DIR / "cardiff",
    ),
    "siebert": SentimentModelConfig(
        key="siebert",
        display_name="SieBERT RoBERTa Large",
        hf_id="siebert/sentiment-roberta-large-english",
        checkpoint_dir=MODELS_DIR / "siebert",
    ),
}

# SieBERT is binary (Positive/Negative). A synthetic Neutral probability of
# ``NEUTRAL_UNCERTAINTY_SCALE * min(p_neg, p_pos)`` is inserted before the
# row-wise renormalisation, so near-ties between the two poles are read as
# Neutral. 2.0 means: predict Neutral when neither pole reaches ~2/3.
NEUTRAL_UNCERTAINTY_SCALE: float = 2.0

# --------------------------------------------------------------------------- #
# Shared inference / reproducibility settings
# --------------------------------------------------------------------------- #
SEED: int = 42
BATCH_SIZE: int = int(os.getenv("SENTIMENT_BATCH_SIZE", "16"))
MAX_LENGTH: int = int(os.getenv("SENTIMENT_MAX_LENGTH", "128"))
DEVICE: str = os.getenv("SENTIMENT_DEVICE", "auto")  # "auto" | "cuda" | "cpu"

# Intra-op thread count for CPU inference. torch's default is the visible core
# count, which is wrong inside a cgroup-limited container (it sees the host's
# cores, oversubscribes, and thrashes) — a common cause of order-of-magnitude
# CPU translation slowdowns. 0 keeps torch's own default; the effective value is
# logged at startup either way so it can be checked against the container limit.
TORCH_NUM_THREADS: int = int(os.getenv("SENTIMENT_TORCH_NUM_THREADS", "0"))

# --------------------------------------------------------------------------- #
# HTTP service limits (api_server.py)
# --------------------------------------------------------------------------- #
# Request-size ceilings. Defaults are far above what the current caller sends
# (one post per request) so they reject only genuinely abusive payloads, never
# ordinary traffic. Requests over a limit get 413 with the limit in the message.
API_MAX_TEXTS: int = int(os.getenv("SENTIMENT_API_MAX_TEXTS", "256"))
API_MAX_TEXT_CHARS: int = int(os.getenv("SENTIMENT_API_MAX_TEXT_CHARS", "5000"))
API_MAX_TOTAL_CHARS: int = int(os.getenv("SENTIMENT_API_MAX_TOTAL_CHARS", "200000"))

# A single request that hangs inside the model (GPU-level stall, pathological
# input) must not take the whole service down with it. Requests queue for the
# inference lock up to this long; past it they fail fast with 503 instead of
# blocking every other request — including /health — forever. Comfortably
# above the slowest observed legitimate single-post translation (~5 min).
INFERENCE_LOCK_TIMEOUT_S: float = float(os.getenv("SENTIMENT_INFERENCE_LOCK_TIMEOUT_S", "600"))
# Once a request owns the inference lock, an uninterruptible model call cannot
# be cancelled safely from another Python thread. Exiting lets PM2 restart the
# process and guarantees the lock is released.
INFERENCE_HARD_TIMEOUT_S: float = float(
    os.getenv("SENTIMENT_INFERENCE_HARD_TIMEOUT_S", "120")
)

# --------------------------------------------------------------------------- #
# Stage 3 — intelligence layer (Ollama)
# --------------------------------------------------------------------------- #
# A SECOND stage that consumes the deterministic pipeline's structured output.
# It never re-does language detection, transliteration, translation or sentiment
# — those are trusted inputs. It adds intent, category, contextual risk,
# reasoning, an executive summary and a recommended action.
#
# The provider is selected by name so further intelligence backends can be
# registered in src/intelligence.py without touching the sentiment pipeline.
INTELLIGENCE_PROVIDER: str = os.getenv("SENTIMENT_INTELLIGENCE_PROVIDER", "ollama")

# Ollama runs on a separate host — set OLLAMA_BASE_URL in .env to point at it.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_S: float = float(os.getenv("OLLAMA_TIMEOUT_S", "120"))
OLLAMA_RETRIES: int = int(os.getenv("OLLAMA_RETRIES", "2"))
# Intelligence calls are network-bound and run outside the model-inference lock,
# so a batch fans out across this many concurrent requests to the Ollama host.
OLLAMA_CONCURRENCY: int = int(os.getenv("OLLAMA_CONCURRENCY", "4"))
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
# Bounds worst-case generation length (summary + intent_label add tokens vs SOCKEYE's
# old 240-token cap). Still small enough that a degenerate decode cannot run away.
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
# Ollama >= 0.5 constrains generation to a JSON schema, which is far more
# reliable than format="json" alone. Set to 0 for older servers; the provider
# also falls back automatically if the server rejects a schema.
OLLAMA_JSON_SCHEMA: bool = os.getenv("OLLAMA_JSON_SCHEMA", "1") not in ("0", "false", "False")

# Taxonomies. "Unknown" is always additionally permitted for category/intent —
# the guard rails require it rather than a guess when evidence is insufficient.
# Callers may replace CATEGORY_LABELS at request time via a policy_pack
# (see src/intelligence.py); these remain the default when no pack is supplied.
INTENT_LABELS: tuple[str, ...] = (
    "Information", "Opinion", "Protest Mobilization", "Call to Action", "Threat",
    "Recruitment", "Rumor", "Propaganda", "Satire", "Misinformation",
)
CATEGORY_LABELS: tuple[str, ...] = (
    "Political", "Religious", "Communal", "Criminal", "Cyber Crime",
    "Hate Speech", "Public Safety", "Protest", "Terrorism", "Fake News",
    "Financial Fraud", "Other",
)
ACTION_LABELS: tuple[str, ...] = (
    "Ignore", "Monitor", "Human Review", "Escalate", "Immediate Attention",
)
UNKNOWN_LABEL: str = "Unknown"

# Edge-case thresholds. These produce deterministic *signals* attached to the
# prompt; the model is told to weigh them. They never decide the verdict, except
# where there is literally no analyzable content (see triage() in intelligence.py).
INTELLIGENCE_MAX_TEXT_CHARS: int = int(os.getenv("SENTIMENT_INTELLIGENCE_MAX_TEXT_CHARS", "4000"))
INTELLIGENCE_LOW_CONFIDENCE: float = float(os.getenv("SENTIMENT_INTELLIGENCE_LOW_CONFIDENCE", "0.60"))
INTELLIGENCE_SHORT_WORDS: int = int(os.getenv("SENTIMENT_INTELLIGENCE_SHORT_WORDS", "3"))
# Fraction of alphabetic characters in the minority script above which a post
# counts as genuinely code-mixed rather than incidental.
INTELLIGENCE_CODEMIX_RATIO: float = float(os.getenv("SENTIMENT_INTELLIGENCE_CODEMIX_RATIO", "0.15"))
# Free-form intent (intent_mode=free) is clamped to this many words server-side.
INTELLIGENCE_INTENT_MAX_WORDS: int = int(os.getenv("SENTIMENT_INTELLIGENCE_INTENT_MAX_WORDS", "8"))
# Hard ceiling on categories accepted in a caller-supplied policy_pack.
INTELLIGENCE_MAX_POLICY_CATEGORIES: int = int(
    os.getenv("SENTIMENT_INTELLIGENCE_MAX_POLICY_CATEGORIES", "128")
)

# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
# ISO 639-1 codes considered by the lingua detector (Indian languages + English).
DETECTOR_LANGUAGES: tuple[str, ...] = (
    "en", "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur",
)
