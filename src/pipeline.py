"""Production pipeline: IndicLID -> IndicXlit -> IndicTrans2/NLLB -> Cardiff.

    post -> preprocessing -> language detection (lingua/langdetect)
         -> Latin-script? refine language with IndicLID-FTR (romanized-Indic
            vs genuine English — lingua/fastText-lid176 can't tell these
            apart, tested 0/7 correct; IndicLID-FTR is trained for exactly
            this and gets ~71%)
         -> romanized Indic? IndicXlit roman -> native script : unchanged
         -> genuine English? bypass : translate with IndicTrans2
         -> Cardiff Twitter RoBERTa
         -> Positive / Neutral / Negative + confidence + timings

NLLB is loaded as a deterministic fallback for failed or unusable primary
translations; it does not replace IndicTrans2 as the authoritative path.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

import config
from src.inference import SentimentClassifier
from src.language_detector import LanguageDetector
from src.lid_roman import RomanLanguageDetector
from src.preprocessing import clean_text
from src.script_detection import (
    extract_latin_text,
    is_latin_script,
    latin_letter_ratio,
    unique_indic_language,
)
from src.transliteration import Transliterator
from src.translation import (
    TranslationError,
    TranslationOutputError,
    TranslationResult,
    Translator,
    translation_is_usable,
)
from src.utils import configure_torch_threads, resolve_device, set_seed

logger = logging.getLogger("benchmark.pipeline")

TRANSLATION_KEY = "indictrans2"
FALLBACK_TRANSLATION_KEY = "nllb"
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
    cleaned_text: str = ""
    transliterated_text: str = ""
    translation_backend: str = "unknown"
    fallback_used: bool = False
    fallback_reason: str = ""
    translation_truncated: bool = False
    sentiment_truncated: bool = False
    low_confidence: bool = False
    review_recommended: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineFailure:
    """Recoverable per-post failure for the API's Ollama fallback."""

    post_text: str
    language: str
    was_transliterated: bool
    deterministic_time_ms: float
    reason: str
    translation_time_ms: float = 0.0


class RecoverablePipelineError(RuntimeError):
    """Typed model-stage failure that may be retried per post."""

    def __init__(self, message: str, translation_time_ms: float = 0.0) -> None:
        super().__init__(message)
        self.translation_time_ms = max(0.0, float(translation_time_ms))


def _raise_if_fatal_model_error(exc: Exception) -> None:
    """Never disguise resource exhaustion or programming/invariant failures."""
    fatal_types = (MemoryError, AssertionError, AttributeError, TypeError)
    if isinstance(exc, fatal_types):
        raise exc
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        raise exc
    message = str(exc).casefold()
    if isinstance(exc, RuntimeError) and any(
        marker in message
        for marker in ("out of memory", "cuda error", "device-side assert")
    ):
        raise exc


def annotate_result_quality(result: PipelineResult) -> PipelineResult:
    """Attach review flags without changing the sentiment label.

    The low-confidence cutoff is the existing intelligence threshold
    (INTELLIGENCE_LOW_CONFIDENCE), not a newly invented number. Flags are
    additive metadata for operators; they never rewrite ``sentiment``.
    """
    low = float(result.confidence) < config.INTELLIGENCE_LOW_CONFIDENCE
    review = (
        low
        or result.language in {"unknown", ""}
        or (result.language != "en" and not result.was_translated)
        or result.fallback_used
        or result.translation_truncated
        or result.sentiment_truncated
    )
    result.low_confidence = low
    result.review_recommended = review
    return result


class SentimentPipeline:
    """Load once, then call :meth:`predict_one` / :meth:`predict_batch`.

    Thread safety: :meth:`predict_batch` holds a pipeline-wide lock for its whole
    body, so concurrent callers queue instead of interleaving. Every stage keeps
    per-instance mutable state (fairseq generators, tokenizer source language,
    torch modules) that is not safe to drive from two threads at once; the
    individual stages lock too, but the outer lock is what keeps a single
    request's stages from being interleaved with another's.
    """

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
        if self.device.type == "cpu":
            configure_torch_threads(config.TORCH_NUM_THREADS)

        self._lock = threading.Lock()
        self._freed = False

        self.detector = LanguageDetector(seed=seed)
        self.roman_detector = RomanLanguageDetector(device=self.device)
        self.transliterator = Transliterator(device=self.device)
        self.translator = Translator(
            config.TRANSLATION_MODELS[TRANSLATION_KEY], self.device
        )
        self.fallback_translator = Translator(
            config.TRANSLATION_MODELS[FALLBACK_TRANSLATION_KEY], self.device
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

    def stage_status(self) -> dict[str, object]:
        """Which stages are actually live, for /health and startup logging.

        Every optional stage degrades silently by design (a missing fasttext or
        fairseq install disables roman LID / transliteration and the pipeline
        keeps going), so a caller seeing plausible-but-wrong output has no way
        to tell a fully-configured deployment from a half-configured one. This
        makes that difference observable without log access.
        """
        return {
            "device": self.device.type,
            "language_detector": self.detector.backend,
            "roman_lid_ftr": self.roman_detector.ftr is not None,
            "roman_lid_bert": self.roman_detector.bert is not None,
            "transliteration_languages": sorted(self.transliterator.models),
            "translator": self.translator.cfg.display_name,
            "translator_preprocessing": self.translator.preprocessing_mode,
            "urdu_preprocessing_ready": self.translator.urdu_preprocessing_ready,
            "fallback_translator": self.fallback_translator.cfg.display_name,
            "generation_max_time_s": config.GENERATION_MAX_TIME_S,
            "inference_hard_timeout_s": config.INFERENCE_HARD_TIMEOUT_S,
            "sentiment_model": self.classifier.cfg.display_name,
        }

    def _refine_latin_languages(self, cleaned: list[str], languages: list[str]) -> list[str]:
        """The general detector (lingua) has no romanized-Indic language
        profiles, so Latin-script text almost always comes back as 'en' even
        when it's Hinglish/Roman-Telugu/etc. IndicLID-FTR is purpose-trained
        for exactly this distinction — use it to override the language for
        Latin-script text, and for mixed-script posts that lingua called
        English/unknown despite a substantial Latin span.
        Native-script detections (te/hi/…) are left alone so a single English
        loanword cannot flip the post to English.
        Unique Indic scripts (Kannada, Malayalam, …) override lingua
        ``en``/``unknown`` because lingua has no KN/ML profiles.
        """
        refined = list(languages)
        for i, (text, lang) in enumerate(zip(cleaned, languages)):
            script_lang = unique_indic_language(text)
            if script_lang is not None and lang in {"en", "unknown"}:
                refined[i] = script_lang
                continue
            probe = None
            if is_latin_script(text):
                probe = text
            elif lang in {"en", "unknown"} and latin_letter_ratio(
                text
            ) >= config.INTELLIGENCE_CODEMIX_RATIO:
                probe = extract_latin_text(text)
            if not probe or not probe.strip():
                continue
            better = self.roman_detector.detect(probe)
            if better is None:
                continue
            if lang not in {"en", "unknown"} and better == "en":
                continue
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

    def _translate_with_fallback(
        self, texts: list[str], languages: list[str], batch_size: int
    ) -> TranslationResult:
        """Use NLLB only when IndicTrans2 fails or emits unusable English."""
        try:
            primary = self.translator.translate(
                texts, languages, batch_size=batch_size
            )
        except Exception as exc:
            _raise_if_fatal_model_error(exc)
            logger.error(
                "Primary translator failed (%s); retrying batch with %s",
                exc,
                self.fallback_translator.cfg.display_name
                if hasattr(self.fallback_translator, "cfg")
                else "fallback translator",
            )
            try:
                fallback = self.fallback_translator.translate(
                    texts, languages, batch_size=batch_size
                )
            except Exception as fallback_exc:
                _raise_if_fatal_model_error(fallback_exc)
                raise TranslationOutputError(
                    "Primary and fallback translators both failed"
                ) from fallback_exc
            self._validate_fallback(texts, languages, fallback)
            return fallback

        retry_indices = [
            i
            for i, (source, lang) in enumerate(zip(texts, languages))
            if lang != "en"
            and lang in config.FLORES_CODES
            and (
                not bool(primary.translated_mask[i])
                or not translation_is_usable(source, primary.texts[i])
            )
        ]
        if not retry_indices:
            return primary

        logger.warning(
            "Primary translation rejected for %d post(s); retrying with %s",
            len(retry_indices),
            self.fallback_translator.cfg.display_name
            if hasattr(self.fallback_translator, "cfg")
            else "fallback translator",
        )
        retry_texts = [texts[i] for i in retry_indices]
        retry_languages = [languages[i] for i in retry_indices]
        try:
            fallback = self.fallback_translator.translate(
                retry_texts, retry_languages, batch_size=batch_size
            )
        except Exception as fallback_exc:
            _raise_if_fatal_model_error(fallback_exc)
            raise TranslationOutputError(
                "Fallback translator failed for rejected primary output"
            ) from fallback_exc
        self._validate_fallback(retry_texts, retry_languages, fallback)

        merged_texts = list(primary.texts)
        merged_times = primary.times_ms.copy()
        merged_mask = primary.translated_mask.copy()
        merged_backends = list(
            primary.backends or ["unknown"] * len(primary.texts)
        )
        merged_truncated = (
            primary.truncated_mask.copy()
            if primary.truncated_mask is not None
            else np.zeros(len(primary.texts), dtype=bool)
        )
        fallback_backends = fallback.backends or [FALLBACK_TRANSLATION_KEY] * len(
            retry_indices
        )
        fallback_truncated = fallback.truncated_mask
        for fallback_idx, original_idx in enumerate(retry_indices):
            merged_texts[original_idx] = fallback.texts[fallback_idx]
            merged_times[original_idx] += fallback.times_ms[fallback_idx]
            merged_mask[original_idx] = fallback.translated_mask[fallback_idx]
            merged_backends[original_idx] = fallback_backends[fallback_idx]
            if fallback_truncated is not None:
                merged_truncated[original_idx] = bool(
                    fallback_truncated[fallback_idx]
                )
        return TranslationResult(
            texts=merged_texts,
            times_ms=merged_times,
            total_time_s=primary.total_time_s + fallback.total_time_s,
            gpu_peak_mb=max(primary.gpu_peak_mb, fallback.gpu_peak_mb),
            translated_mask=merged_mask,
            n_cached=primary.n_cached + fallback.n_cached,
            backends=merged_backends,
            truncated_mask=merged_truncated,
        )

    @staticmethod
    def _validate_fallback(
        texts: list[str], languages: list[str], result: TranslationResult
    ) -> None:
        bad = [
            i
            for i, (source, lang) in enumerate(zip(texts, languages))
            if lang != "en"
            and lang in config.FLORES_CODES
            and (
                not bool(result.translated_mask[i])
                or not translation_is_usable(source, result.texts[i])
            )
        ]
        if bad:
            raise TranslationOutputError(
                f"Fallback translator returned unusable output for {len(bad)} post(s)"
            )

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = config.BATCH_SIZE,
        translation_batch_size: int = config.TRANSLATION_BATCH_SIZE,
        request_id: object = None,
    ) -> list[PipelineResult]:
        """Compatibility wrapper that requires every deterministic item to pass."""
        outcomes = self.predict_batch_outcomes(
            texts,
            batch_size=batch_size,
            translation_batch_size=translation_batch_size,
            request_id=request_id,
        )
        failures = [
            outcome for outcome in outcomes if isinstance(outcome, PipelineFailure)
        ]
        if failures:
            raise TranslationOutputError(
                f"Deterministic pipeline failed for {len(failures)} post(s)"
            )
        return outcomes

    def predict_batch_outcomes(
        self,
        texts: list[str],
        batch_size: int = config.BATCH_SIZE,
        translation_batch_size: int = config.TRANSLATION_BATCH_SIZE,
        request_id: object = None,
    ) -> list[PipelineResult | PipelineFailure]:
        """Run a batch, isolating recoverable errors to individual posts.

        Serialized on the pipeline lock — see the class docstring. `request_id`
        is optional and purely for log correlation with the caller's own
        request log (e.g. api_server.py's "req N: received" line) — pass it
        through when available so a hang's last logged checkpoint says which
        request it was, not just which stage.
        """
        with self._lock:
            try:
                results = self._predict_batch_locked(
                    texts, batch_size, translation_batch_size, request_id
                )
                return self._validated_outcomes(results)
            except (TranslationError, RecoverablePipelineError) as batch_exc:
                logger.warning(
                    "req %s: deterministic batch failed (%s); isolating %d post(s)",
                    request_id,
                    type(batch_exc).__name__,
                    len(texts),
                )

            outcomes: list[PipelineResult | PipelineFailure] = []
            for index, text in enumerate(texts):
                started = time.perf_counter()
                try:
                    outcomes.extend(
                        self._validated_outcomes(
                            self._predict_batch_locked(
                                [text],
                                1,
                                1,
                                f"{request_id}.{index}",
                            )
                        )
                    )
                except (TranslationError, RecoverablePipelineError) as exc:
                    language, was_transliterated = self._failure_context(text)
                    outcomes.append(
                        PipelineFailure(
                            post_text=text,
                            language=language,
                            was_transliterated=was_transliterated,
                            deterministic_time_ms=round(
                                (time.perf_counter() - started) * 1000.0, 3
                            ),
                            reason=type(exc).__name__,
                            translation_time_ms=round(
                                float(
                                    getattr(exc, "translation_time_ms", 0.0)
                                ),
                                3,
                            ),
                        )
                    )
            return outcomes

    @staticmethod
    def _validated_outcomes(
        results: list[PipelineResult],
    ) -> list[PipelineResult | PipelineFailure]:
        """Convert malformed deterministic records into fallback candidates."""
        outcomes: list[PipelineResult | PipelineFailure] = []
        for result in results:
            timings = (
                result.translation_time_ms,
                result.sentiment_time_ms,
                result.total_time_ms,
            )
            valid = (
                isinstance(result.english_text, str)
                and bool(result.english_text.strip())
                and result.sentiment in {"Positive", "Neutral", "Negative"}
                and not isinstance(result.confidence, bool)
                and isinstance(result.confidence, (int, float))
                and math.isfinite(float(result.confidence))
                and 0.0 <= float(result.confidence) <= 1.0
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                    for value in timings
                )
                and abs(
                    float(result.total_time_ms)
                    - float(
                        result.translation_time_ms
                        + result.sentiment_time_ms
                    )
                )
                <= 0.01
            )
            if valid:
                outcomes.append(result)
                continue
            outcomes.append(
                PipelineFailure(
                    post_text=result.post_text,
                    language=result.language,
                    was_transliterated=result.was_transliterated,
                    deterministic_time_ms=max(
                        0.0,
                        float(result.total_time_ms)
                        if isinstance(result.total_time_ms, (int, float))
                        and math.isfinite(float(result.total_time_ms))
                        else 0.0,
                    ),
                    reason="InvalidPipelineResult",
                    translation_time_ms=max(
                        0.0,
                        float(result.translation_time_ms)
                        if isinstance(
                            result.translation_time_ms, (int, float)
                        )
                        and math.isfinite(float(result.translation_time_ms))
                        else 0.0,
                    ),
                )
            )
        return outcomes

    def _failure_context(self, text: str) -> tuple[str, bool]:
        """Best-effort metadata for a failed item; never mask its root failure."""
        try:
            cleaned = clean_text(text)
            languages = self.detector.detect_batch([cleaned])
            languages = self._refine_latin_languages([cleaned], languages)
            _working, transliterated = self._transliterate_batch(
                [cleaned], languages
            )
            return languages[0], transliterated[0]
        except Exception as exc:
            logger.warning(
                "Could not recover failed-post language metadata (%s)",
                type(exc).__name__,
            )
            return "unknown", False

    def _predict_batch_locked(
        self, texts: list[str], batch_size: int, translation_batch_size: int,
        request_id: object = None,
    ) -> list[PipelineResult]:
        if self._freed:
            raise RuntimeError("SentimentPipeline.free() has been called; reload the pipeline.")

        # Stage checkpoints: logged as each stage starts, not just in the
        # summary line at the end. A hang inside any one stage means the
        # summary line never fires — without these, the logs show a request
        # was "received" and then nothing, with no way to tell which of the
        # five stages below it died in. The last "stage=" line before a hang
        # names the culprit.
        tag = f"req {request_id}" if request_id is not None else "batch"

        # Per-stage wall-clock, logged at the end of every batch. The result
        # objects only carry translation and sentiment timings, so without this
        # a slow request gives no way to tell which stage actually cost the time.
        t_start = time.perf_counter()
        logger.info("%s: stage=detect (%d text(s))", tag, len(texts))
        cleaned = [clean_text(text) for text in texts]
        languages = self.detector.detect_batch(cleaned)
        t_detect = time.perf_counter()
        logger.info("%s: stage=roman_lid", tag)
        languages = self._refine_latin_languages(cleaned, languages)
        t_refine = time.perf_counter()
        logger.info("%s: stage=transliterate", tag)
        working_texts, was_transliterated = self._transliterate_batch(cleaned, languages)
        t_xlit = time.perf_counter()

        logger.info("%s: stage=translate", tag)
        translation_started = time.perf_counter()
        try:
            translation = self._translate_with_fallback(
                working_texts, languages, translation_batch_size
            )
        except TranslationError as exc:
            raise RecoverablePipelineError(
                "Deterministic translation failed",
                translation_time_ms=(
                    time.perf_counter() - translation_started
                )
                * 1000.0,
            ) from exc
        t_translate = time.perf_counter()
        logger.info("%s: stage=sentiment", tag)
        try:
            sentiment = self.classifier.predict(
                translation.texts, batch_size=batch_size
            )
        except Exception as exc:
            _raise_if_fatal_model_error(exc)
            translation_ms = (
                float(translation.times_ms[0])
                if len(texts) == 1 and len(translation.times_ms) == 1
                else 0.0
            )
            raise RecoverablePipelineError(
                "Deterministic sentiment failed",
                translation_time_ms=translation_ms,
            ) from exc
        t_end = time.perf_counter()
        logger.info("%s: stage=done", tag)

        logger.info(
            "Batch of %d done in %.0f ms — detect %.0f | roman-lid %.0f | "
            "translit %.0f (%d posts) | translate %.0f (%d posts, %d cached) | "
            "sentiment %.0f | languages %s",
            len(texts), (t_end - t_start) * 1000.0,
            (t_detect - t_start) * 1000.0,
            (t_refine - t_detect) * 1000.0,
            (t_xlit - t_refine) * 1000.0, sum(was_transliterated),
            (t_translate - t_xlit) * 1000.0,
            int(translation.translated_mask.sum()), translation.n_cached,
            (t_end - t_translate) * 1000.0,
            sorted(set(languages)),
        )

        backends = translation.backends or ["unknown"] * len(texts)
        truncated = (
            translation.truncated_mask
            if translation.truncated_mask is not None
            else np.zeros(len(texts), dtype=bool)
        )
        return [
            annotate_result_quality(
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
                    cleaned_text=cleaned[i],
                    transliterated_text=(
                        working_texts[i] if was_transliterated[i] else ""
                    ),
                    translation_backend=backends[i],
                    fallback_used=False,
                    fallback_reason="",
                    translation_truncated=bool(truncated[i]),
                    sentiment_truncated=self._sentiment_input_truncated(
                        translation.texts[i]
                    ),
                )
            )
            for i in range(len(texts))
        ]

    def _sentiment_input_truncated(self, text: str) -> bool:
        try:
            encoded = self.classifier.tokenizer(
                text, add_special_tokens=True, truncation=False
            )
            return len(encoded["input_ids"]) > self.classifier.max_length
        except Exception:
            return False

    def predict_one(self, text: str) -> PipelineResult:
        """Convenience wrapper for a single post."""
        return self.predict_batch([text], batch_size=1, translation_batch_size=1)[0]

    def free(self) -> None:
        """Release every loaded model. Idempotent.

        Includes the roman-LID stage, which holds the largest single artifact in
        the pipeline (IndicLID-BERT, ~1.1 GB) — it used to be left resident here
        while the smaller models were released.
        """
        with self._lock:
            if self._freed:
                return
            self._freed = True
            self.roman_detector.free()
            self.transliterator.free()
            self.translator.free()
            self.fallback_translator.free()
            self.classifier.free()
            logger.info("Pipeline models released.")
