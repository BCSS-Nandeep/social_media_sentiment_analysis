"""Finalized production pipeline: IndicTrans2 -> Cardiff Twitter RoBERTa.

This is the sole end-to-end architecture selected by the benchmark:

    post -> preprocessing -> language detection
         -> (English? bypass : translate with IndicTrans2)
         -> Cardiff Twitter RoBERTa
         -> Positive / Neutral / Negative + confidence + timings

No other translation or sentiment models are loaded here.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch

import config
from src.inference import SentimentClassifier
from src.language_detector import LanguageDetector
from src.preprocessing import clean_text
from src.translation import Translator
from src.utils import resolve_device, set_seed

logger = logging.getLogger("benchmark.pipeline")

TRANSLATION_KEY = "indictrans2"
SENTIMENT_KEY = "cardiff"


@dataclass
class PipelineResult:
    """Everything the pipeline returns for one post."""

    post_text: str            # original input
    language: str             # detected ISO 639-1 code (or 'unknown')
    english_text: str         # translated text, or the cleaned input if bypassed
    was_translated: bool      # False for English / unmapped-language bypass
    sentiment: str            # Positive | Neutral | Negative
    confidence: float         # probability of the predicted class
    translation_time_ms: float
    sentiment_time_ms: float
    total_time_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class SentimentPipeline:
    """Load once, then call :meth:`predict_one` / :meth:`predict_batch`."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        seed: int = config.SEED,
        max_length: int = config.MAX_LENGTH,
    ) -> None:
        set_seed(seed)
        if device is None or isinstance(device, str):
            self.device = resolve_device(device or config.DEVICE)
        else:
            self.device = device

        self.detector = LanguageDetector(seed=seed)
        self.translator = Translator(
            config.TRANSLATION_MODELS[TRANSLATION_KEY], self.device
        )
        self.classifier = SentimentClassifier(
            config.SENTIMENT_MODELS[SENTIMENT_KEY], self.device, max_length=max_length
        )
        logger.info(
            "Production pipeline ready: %s -> %s on %s",
            self.translator.cfg.display_name,
            self.classifier.cfg.display_name,
            self.device.type,
        )

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = config.BATCH_SIZE,
        translation_batch_size: int = config.TRANSLATION_BATCH_SIZE,
    ) -> list[PipelineResult]:
        """Run the full pipeline over a list of posts (order preserved)."""
        cleaned = [clean_text(text) for text in texts]
        languages = self.detector.detect_batch(cleaned)

        translation = self.translator.translate(
            cleaned, languages, batch_size=translation_batch_size
        )
        sentiment = self.classifier.predict(translation.texts, batch_size=batch_size)

        return [
            PipelineResult(
                post_text=texts[i],
                language=languages[i],
                english_text=translation.texts[i],
                was_translated=bool(translation.translated_mask[i]),
                sentiment=sentiment.labels[i],
                confidence=round(float(sentiment.confidences[i]), 4),
                translation_time_ms=round(float(translation.times_ms[i]), 3),
                sentiment_time_ms=round(float(sentiment.times_ms[i]), 3),
                total_time_ms=round(
                    float(translation.times_ms[i] + sentiment.times_ms[i]), 3
                ),
            )
            for i in range(len(texts))
        ]

    def predict_one(self, text: str) -> PipelineResult:
        """Convenience wrapper for a single post."""
        return self.predict_batch([text], batch_size=1, translation_batch_size=1)[0]

    def free(self) -> None:
        """Release both models."""
        self.translator.free()
        self.classifier.free()
