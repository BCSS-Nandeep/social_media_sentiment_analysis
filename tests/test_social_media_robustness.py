"""Regression tests for general social-media robustness (no GPU models)."""
from __future__ import annotations

import unittest

from src import pipeline
from src import script_detection
from src import transliteration
from src.preprocessing import clean_text
from src.transliteration import Transliterator


class PreprocessingRobustnessTests(unittest.TestCase):
    def test_emojis_are_preserved(self):
        self.assertIn("😡", clean_text("Power cuts every day 😡"))

    def test_urls_and_mentions_are_removed_hashtags_kept_as_words(self):
        cleaned = clean_text("See https://example.com @officer #Disappointed really")
        self.assertNotIn("http", cleaned)
        self.assertNotIn("@officer", cleaned)
        self.assertIn("Disappointed", cleaned)
        self.assertNotIn("#", cleaned)

    def test_repeated_whitespace_collapses(self):
        self.assertEqual("a b", clean_text("a    b"))


class ScriptAndLatinExtractionTests(unittest.TestCase):
    def test_mixed_script_is_not_pure_latin(self):
        text = "Good scheme ప్రభుత్వం చాలా బాగుంది"
        self.assertFalse(script_detection.is_latin_script(text))
        self.assertGreaterEqual(script_detection.latin_letter_ratio(text), 0.15)

    def test_latin_span_extracts_roman_tokens(self):
        span = script_detection.extract_latin_text(
            "Ee scheme valla ma village lo help ayyindi"
        )
        self.assertIn("scheme", span.casefold())
        self.assertIn("village", span.casefold())

    def test_pure_telugu_has_zero_latin_ratio(self):
        self.assertEqual(
            0.0,
            script_detection.latin_letter_ratio("ప్రభుత్వం ప్రకటించిన కొత్త పథకం"),
        )


class EnglishPreserveTransliterationTests(unittest.TestCase):
    def test_should_preserve_common_english_and_acronyms(self):
        self.assertTrue(transliteration.should_preserve_roman("Good"))
        self.assertTrue(transliteration.should_preserve_roman("disappointing"))
        self.assertTrue(transliteration.should_preserve_roman("TDP"))
        self.assertTrue(transliteration.should_preserve_roman("scheme"))
        self.assertFalse(transliteration.should_preserve_roman("ayyindi"))
        self.assertFalse(transliteration.should_preserve_roman("chala"))
        self.assertFalse(transliteration.should_preserve_roman("nenu"))

    def test_transliterate_skips_english_tokens(self):
        class FakeModel:
            def translate(self, prepared, beam):
                self.prepared = prepared
                return ["అయ్యింది"] * len(prepared)

        service = Transliterator.__new__(Transliterator)
        service.models = {"te": FakeModel()}

        result = service.transliterate(
            "Ee scheme valla help ayyindi Good move", "te"
        )

        self.assertIsNotNone(result)
        self.assertIn("scheme", result)
        self.assertIn("help", result)
        self.assertIn("Good", result)
        self.assertIn("move", result)
        self.assertNotIn("__te__ s c h e m e", " ".join(service.models["te"].prepared))


class MixedScriptLanguageRefineTests(unittest.TestCase):
    def test_english_misdetect_on_mixed_script_can_be_refined(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)

        class FakeRoman:
            def detect(self, text):
                self.seen = text
                return "te"

        service.roman_detector = FakeRoman()
        languages = service._refine_latin_languages(
            ["Good scheme ప్రభుత్వం చాలా బాగుంది"],
            ["en"],
        )
        self.assertEqual(["te"], languages)
        self.assertTrue(script_detection.is_latin_script(service.roman_detector.seen))

    def test_native_telugu_detection_is_not_overridden_by_english_loanword(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)

        class FakeRoman:
            def detect(self, text):
                self.called = True
                return "en"

        detector = FakeRoman()
        detector.called = False
        service.roman_detector = detector
        languages = service._refine_latin_languages(
            ["ప్రభుత్వం ప్రకటించిన కొత్త పథకం scheme"],
            ["te"],
        )
        self.assertEqual(["te"], languages)


class TraceabilityAndFallbackMetadataTests(unittest.TestCase):
    def test_pipeline_result_exposes_additive_trace_fields(self):
        result = pipeline.PipelineResult(
            post_text="raw",
            language="te",
            english_text="The scheme is good",
            was_translated=True,
            was_transliterated=True,
            sentiment="Positive",
            confidence=0.91,
            translation_time_ms=10.0,
            sentiment_time_ms=2.0,
            total_time_ms=12.0,
            cleaned_text="raw",
            transliterated_text="native",
            translation_backend="indictrans2",
            fallback_used=False,
            fallback_reason="",
            translation_truncated=False,
            sentiment_truncated=False,
        )
        payload = result.to_dict()
        self.assertEqual("raw", payload["post_text"])
        self.assertEqual("native", payload["transliterated_text"])
        self.assertEqual("indictrans2", payload["translation_backend"])
        self.assertIn("english_text", payload)
        self.assertFalse(payload["fallback_used"])
        self.assertTrue(payload["low_confidence"] is False or payload["confidence"] >= 0.60)

    def test_low_confidence_does_not_change_sentiment_label(self):
        result = pipeline.PipelineResult(
            post_text="ok",
            language="en",
            english_text="ok",
            was_translated=False,
            was_transliterated=False,
            sentiment="Neutral",
            confidence=0.41,
            translation_time_ms=0.0,
            sentiment_time_ms=1.0,
            total_time_ms=1.0,
        )
        finalized = pipeline.annotate_result_quality(result)
        self.assertEqual("Neutral", finalized.sentiment)
        self.assertTrue(finalized.low_confidence)
        self.assertTrue(finalized.review_recommended)

    def test_ollama_fallback_marks_backend_without_dropping_original(self):
        from src.pipeline_fallback import OllamaPipelineFallback

        class FakeProvider:
            def generate(self, payload, **kwargs):
                return {
                    "english_text": "Whoever has not performed the prayer",
                    "sentiment": "Neutral",
                    "confidence": 0.82,
                }

            def describe(self):
                return {"provider": "ollama"}

        failure = pipeline.PipelineFailure(
            post_text="Jis kisi ki namaz nahi hui",
            language="ur",
            was_transliterated=True,
            deterministic_time_ms=12.5,
            reason="TranslationOutputError",
            translation_time_ms=7.5,
        )
        result = OllamaPipelineFallback(
            provider=FakeProvider(), timeout_s=5
        ).resolve(failure)
        self.assertEqual("Jis kisi ki namaz nahi hui", result.post_text)
        self.assertTrue(result.fallback_used)
        self.assertEqual("ollama_fallback", result.translation_backend)
        self.assertEqual("TranslationOutputError", result.fallback_reason)
        self.assertEqual("Neutral", result.sentiment)


class IntelligenceDoesNotRewriteSentimentTests(unittest.TestCase):
    def test_payload_includes_trace_fields_and_keeps_sentiment(self):
        from src.intelligence import build_payload

        payload = build_payload(
            {
                "post_text": "original",
                "english_text": "translated",
                "language": "te",
                "sentiment": "Negative",
                "confidence": 0.93,
                "was_translated": True,
                "was_transliterated": False,
                "translation_backend": "indictrans2",
                "fallback_used": False,
            },
            ["code_mixed"],
        )
        self.assertEqual("Negative", payload["sentiment"])
        self.assertEqual("original", payload["original_text"])
        self.assertEqual("translated", payload["english_text"])
        self.assertEqual("indictrans2", payload["translation_backend"])
        self.assertFalse(payload["fallback_used"])


if __name__ == "__main__":
    unittest.main()
