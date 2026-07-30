"""Language detection with a lingua -> langdetect -> 'unknown' fallback chain."""
from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from config import DETECTOR_LANGUAGES

logger = logging.getLogger("benchmark.language")

UNKNOWN = "unknown"


class LanguageDetector:
    """Detect the language of short social-media posts.

    Prefers ``lingua`` (more reliable on short, code-mixed text); falls back to
    ``langdetect`` if lingua is unavailable, and to ``'unknown'`` when neither
    backend can produce a verdict.
    """

    def __init__(self, seed: int = 42) -> None:
        self.backend: str = "none"
        self._lingua = None
        self._langdetect = None

        try:
            from lingua import IsoCode639_1, LanguageDetectorBuilder

            iso_codes = [
                getattr(IsoCode639_1, code.upper())
                for code in DETECTOR_LANGUAGES
                if hasattr(IsoCode639_1, code.upper())
            ]
            self._lingua = (
                LanguageDetectorBuilder.from_iso_codes_639_1(*iso_codes)
                .with_minimum_relative_distance(0.10)
                .build()
            )
            self.backend = "lingua"
            logger.info("Language detection backend: lingua (%d languages)", len(iso_codes))
            return
        except Exception as exc:  # ImportError or model-load failure
            logger.warning("lingua unavailable (%s); trying langdetect.", exc)

        try:
            import langdetect
            from langdetect import DetectorFactory

            DetectorFactory.seed = seed  # make langdetect deterministic
            self._langdetect = langdetect
            self.backend = "langdetect"
            logger.info("Language detection backend: langdetect")
        except Exception as exc:
            logger.error("No language-detection backend available (%s).", exc)

    def detect(self, text: str) -> str:
        """Return the ISO 639-1 code for *text*, or ``'unknown'``."""
        if not text or not text.strip():
            return UNKNOWN
        try:
            if self.backend == "lingua":
                lang = self._lingua.detect_language_of(text)
                if lang is None:
                    return UNKNOWN
                return lang.iso_code_639_1.name.lower()
            if self.backend == "langdetect":
                code: Optional[str] = self._langdetect.detect(text)
                return code.lower() if code else UNKNOWN
        except Exception:
            return UNKNOWN
        return UNKNOWN

    def detect_batch(self, texts: list[str]) -> list[str]:
        """Detect languages for a list of posts with a progress bar."""
        return [
            self.detect(text)
            for text in tqdm(texts, desc="Detecting languages", unit="post")
        ]
