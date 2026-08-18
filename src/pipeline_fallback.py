"""Schema-compatible Ollama fallback for recoverable pipeline failures."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import config
from src.intelligence import OllamaProvider
from src.ollama_gate import OllamaCircuitOpen, OllamaGateFull
from src.pipeline import PipelineFailure, PipelineResult
from src.translation import translation_is_usable


ALLOWED_SENTIMENTS = frozenset(("Positive", "Neutral", "Negative"))
EXPECTED_FIELDS = frozenset(("english_text", "sentiment", "confidence"))

FALLBACK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["english_text", "sentiment", "confidence"],
    "properties": {
        "english_text": {"type": "string", "minLength": 1},
        "sentiment": {
            "type": "string",
            "enum": ["Positive", "Neutral", "Negative"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

FALLBACK_SYSTEM_PROMPT = """You are a strict fallback component inside a fixed
translation and sentiment pipeline. The detected_language value is
authoritative. For example, detected_language "ur" means Urdu even when the
post uses Latin letters (Roman Urdu). Translate every source-language phrase
into clear, idiomatic English. Never transliterate, romanize, or paraphrase in
the source language. Preserve proper names, dates, URLs, hashtags, religious
names normally retained in English, and quoted meaning. Examples: Roman Urdu
"mujhe yeh pasand nahi" becomes "I do not like this"; "kal namaz hui" becomes
"the prayer took place yesterday". Then classify only its sentiment.

Return one JSON object with exactly these fields:
- english_text: faithful English translation
- sentiment: exactly Positive, Neutral, or Negative
- confidence: number from 0.0 through 1.0

Do not perform policy analysis, risk scoring, entity extraction, summarization,
alerting, recommendations, or any other task. Do not add fields."""


class PipelineFallbackError(RuntimeError):
    """Ollama could not produce a contract-safe pipeline result."""


class OllamaPipelineFallback:
    def __init__(
        self,
        provider: OllamaProvider | None = None,
        timeout_s: float = config.PIPELINE_FALLBACK_TIMEOUT_S,
        english_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self.provider = provider or OllamaProvider(
            timeout_s=timeout_s,
            retries=0,
        )
        self.timeout_s = float(timeout_s)
        self.english_validator = english_validator

    def describe(self) -> dict:
        provider = self.provider.describe()
        return {
            "enabled": config.PIPELINE_FALLBACK_ENABLED,
            "provider": provider.get("provider", "ollama"),
            "base_url": provider.get("base_url"),
            "model": provider.get("model"),
            "json_schema": provider.get("json_schema", True),
            "timeout_s": self.timeout_s,
            "max_workers": config.PIPELINE_FALLBACK_MAX_WORKERS,
        }

    def health(self) -> bool:
        return self.provider.health()

    def close(self) -> None:
        self.provider.close()

    def resolve(self, failure: PipelineFailure) -> PipelineResult:
        started = time.perf_counter()
        try:
            response = self.provider.generate(
                {
                    "post_text": failure.post_text,
                    "detected_language": failure.language,
                },
                system_prompt=FALLBACK_SYSTEM_PROMPT,
                schema=FALLBACK_SCHEMA,
                timeout_s=self.timeout_s,
            )
            english_text, sentiment, confidence = self._validate(
                response, failure
            )
        except (PipelineFallbackError, OllamaGateFull, OllamaCircuitOpen):
            raise
        except Exception as exc:
            raise PipelineFallbackError(
                f"Ollama pipeline fallback failed: {type(exc).__name__}"
            ) from exc

        fallback_ms = (time.perf_counter() - started) * 1000.0
        translation_ms = round(
            failure.translation_time_ms + fallback_ms, 3
        )
        changed = (
            english_text.strip().casefold()
            != failure.post_text.strip().casefold()
        )
        return PipelineResult(
            post_text=failure.post_text,
            language=failure.language,
            english_text=english_text,
            was_translated=failure.language != "en" and changed,
            was_transliterated=failure.was_transliterated,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            translation_time_ms=translation_ms,
            sentiment_time_ms=0.0,
            total_time_ms=translation_ms,
        )

    def _validate(
        self,
        response: object, failure: PipelineFailure
    ) -> tuple[str, str, float]:
        if not isinstance(response, dict):
            raise PipelineFallbackError("Ollama fallback response is not an object")
        if frozenset(response) != EXPECTED_FIELDS:
            raise PipelineFallbackError(
                "Ollama fallback response fields do not match the contract"
            )

        english_text = response["english_text"]
        sentiment = response["sentiment"]
        confidence = response["confidence"]
        if not isinstance(english_text, str) or not english_text.strip():
            raise PipelineFallbackError("Ollama fallback translation is empty")
        english_text = english_text.strip()
        if sentiment not in ALLOWED_SENTIMENTS:
            raise PipelineFallbackError("Ollama fallback sentiment is invalid")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise PipelineFallbackError("Ollama fallback confidence is invalid")
        if failure.language != "en" and not translation_is_usable(
            failure.post_text, english_text
        ):
            raise PipelineFallbackError("Ollama fallback translation is unusable")
        if (
            failure.language != "en"
            and self.english_validator is not None
            and not self.english_validator(english_text)
        ):
            raise PipelineFallbackError(
                "Ollama fallback output was not verified as English"
            )
        return english_text, sentiment, float(confidence)
