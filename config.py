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

from dataclasses import dataclass
from pathlib import Path

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

TRANSLATION_BATCH_SIZE: int = 8
TRANSLATION_MAX_LENGTH: int = 256
TRANSLATION_NUM_BEAMS: int = 1  # raise to 4-5 for higher quality (slower on CPU)

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
BATCH_SIZE: int = 16
MAX_LENGTH: int = 128
DEVICE: str = "auto"  # "auto" | "cuda" | "cpu"

# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
# ISO 639-1 codes considered by the lingua detector (Indian languages + English).
DETECTOR_LANGUAGES: tuple[str, ...] = (
    "en", "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur",
)
