"""Finalized production pipeline: IndicLID -> IndicXlit -> IndicTrans2 -> Cardiff Twitter RoBERTa.

    post -> preprocessing -> language detection (lingua/langdetect)
         -> Latin-script? refine language with IndicLID-FTR (romanized-Indic
            vs genuine English — lingua/fastText-lid176 can't tell these
            apart, tested 0/7 correct; IndicLID-FTR is trained for exactly
            this and gets ~71%)
         -> romanized Indic? IndicXlit roman -> native script : unchanged
         -> genuine English? bypass : translate with IndicTrans2
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
from src.lid_roman import RomanLanguageDetector
from src.preprocessing import clean_text
from src.script_detection import is_latin_script
from src.transliteration import Transliterator
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
    was_translated: bool      # False only when the detected language has no FLORES mapping, or genuine English
    was_transliterated: bool  # True when IndicXlit converted romanized Indic text to native script first
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
        self.roman_detector = RomanLanguageDetector(device=self.device)
        self.transliterator = Transliterator(device=self.device)
        self.translator = Translator(
            config.TRANSLATION_MODELS[TRANSLATION_KEY], self.device
        )
        self.classifier = SentimentClassifier(
            config.SENTIMENT_MODELS[SENTIMENT_KEY], self.device, max_length=max_length
        )
        logger.info(
            "Production pipeline ready: IndicXlit (%d langs) -> %s -> %s on %s",
            len(self.transliterator.models),
            self.translator.cfg.display_name,
            self.classifier.cfg.display_name,
            self.device.type,
        )

    def _refine_latin_languages(self, cleaned: list[str], languages: list[str]) -> list[str]:
        """The general detector (lingua) has no romanized-Indic language
        profiles, so Latin-script text almost always comes back as 'en' even
        when it's Hinglish/Roman-Telugu/etc. IndicLID-FTR is purpose-trained
        for exactly this distinction — use it to override the language for
        Latin-script text only; native-script detection is untouched."""
        refined = list(languages)
        for i, (text, lang) in enumerate(zip(cleaned, languages)):
            if not is_latin_script(text):
                continue
            better = self.roman_detector.detect(text)
            if better is not None:
                refined[i] = better
        return refined

    def _transliterate_batch(self, cleaned: list[str], languages: list[str]) -> tuple[list[str], list[bool]]:
        """Romanized Indic text (Latin script, Indic language) -> native script
        via IndicXlit, before translation. Native-script text and genuine
        English pass through unchanged. Any transliteration failure falls
        back to the original (romanized) text — same as pre-enhancement
        behavior, just without the accuracy improvement for that one post."""
        working_texts = list(cleaned)
        was_transliterated = [False] * len(cleaned)
        for i, (text, lang) in enumerate(zip(cleaned, languages)):
            if lang == "en" or not text.strip():
                continue
            if not self.transliterator.supports(lang):
                continue
            if not is_latin_script(text):
                continue  # already native script
            result = self.transliterator.transliterate(text, lang)
            if result is not None:
                working_texts[i] = result
                was_transliterated[i] = True
        return working_texts, was_transliterated

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = config.BATCH_SIZE,
        translation_batch_size: int = config.TRANSLATION_BATCH_SIZE,
    ) -> list[PipelineResult]:
        """Run the full pipeline over a list of posts (order preserved)."""
        cleaned = [clean_text(text) for text in texts]
        languages = self.detector.detect_batch(cleaned)
        languages = self._refine_latin_languages(cleaned, languages)
        working_texts, was_transliterated = self._transliterate_batch(cleaned, languages)

        translation = self.translator.translate(
            working_texts, languages, batch_size=translation_batch_size
        )
        sentiment = self.classifier.predict(translation.texts, batch_size=batch_size)

        return [
            PipelineResult(
                post_text=texts[i],
                language=languages[i],
                english_text=translation.texts[i],
                was_translated=bool(translation.translated_mask[i]),
                was_transliterated=was_transliterated[i],
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
        """Release all three models."""
        self.transliterator.free()
        self.translator.free()
        self.classifier.free()
