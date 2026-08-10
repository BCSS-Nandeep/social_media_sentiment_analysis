from __future__ import annotations

import importlib
import importlib.util
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

from src import pipeline
from src import lid_roman
from src import transliteration
from src import translation


ROOT = Path(__file__).resolve().parents[1]


class BrokenProcessor:
    def preprocess_batch(self, *_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'urduhack'")


def translation_result(text: str, translated: bool = True):
    return translation.TranslationResult(
        texts=[text],
        times_ms=np.array([1.0]),
        total_time_s=0.001,
        gpu_peak_mb=0.0,
        translated_mask=np.array([translated]),
    )


class FakeTranslator:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def translate(self, _texts, _languages, batch_size=8):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class UrduDependencyTests(unittest.TestCase):
    def test_requirements_use_only_tensorflow_free_indic_nlp_fork(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("indic-nlp-library-itt==0.1.1", requirements)
        self.assertNotIn("\nindic-nlp-library>=", requirements)

    def test_urdu_preprocessing_failure_is_not_plain_tagged(self):
        translator = translation.Translator.__new__(translation.Translator)
        translator.cfg = types.SimpleNamespace(family="indictrans2")
        translator._indic_processor = BrokenProcessor()
        translator._indic_processor_broken_langs = set()

        with self.assertRaises(Exception) as caught:
            translator._prepare_batch(["جس کسی"], "urd_Arab")

        self.assertEqual("TranslationPreprocessError", type(caught.exception).__name__)

    def test_non_urdu_preprocessing_failure_retains_plain_tag_fallback(self):
        translator = translation.Translator.__new__(translation.Translator)
        translator.cfg = types.SimpleNamespace(family="indictrans2")
        translator._indic_processor = BrokenProcessor()
        translator._indic_processor_broken_langs = set()

        prepared = translator._prepare_batch(["नमस्ते"], "hin_Deva")

        self.assertEqual(["hin_Deva eng_Latn नमस्ते"], prepared)


class RomanLanguageRoutingTests(unittest.TestCase):
    def test_urdu_transliteration_preserves_dates_and_normalizes_common_words(self):
        class FakeModel:
            def translate(self, prepared, beam):
                self.prepared = prepared
                self.beam = beam
                return ["ہ ے"]

        model = FakeModel()
        service = transliteration.Transliterator.__new__(
            transliteration.Transliterator
        )
        service.models = {"ur": model}

        result = service.transliterate("Teen meh 6th Sat hai,", "ur")

        self.assertEqual("تین میں 6th Sat ہے,", result)
        self.assertEqual(["__ur__ h a i"], model.prepared)

    def test_indicxlit_input_includes_language_token_and_lowercase_chars(self):
        self.assertTrue(hasattr(transliteration, "_prepare_word"))

        prepared = transliteration._prepare_word("Namaz", "ur")

        self.assertEqual("__ur__ n a m a z", prepared)

    def test_bert_can_override_confident_ftr_english_for_roman_urdu(self):
        detector = lid_roman.RomanLanguageDetector.__new__(
            lid_roman.RomanLanguageDetector
        )
        detector._ftr_predict = lambda _text: ("eng_Latn", 0.999)
        detector._bert_predict = lambda _text: "urd_Latn"

        language = detector.detect(
            "Jis kissi ki bhi Taraweeh ki Namaz Nahi hui hai"
        )

        self.assertEqual("ur", language)


class TranslationFallbackTests(unittest.TestCase):
    def test_translation_quality_rejects_invalid_outputs(self):
        self.assertTrue(hasattr(translation, "translation_is_usable"))
        self.assertFalse(translation.translation_is_usable("اردو متن", ""))
        self.assertFalse(translation.translation_is_usable("اردو متن", "اردو متن"))
        self.assertFalse(translation.translation_is_usable("اردو متن", "12345 !!!"))
        self.assertFalse(translation.translation_is_usable("اردو متن", "यह अनुवाद नहीं है"))
        self.assertFalse(
            translation.translation_is_usable(
                "اردو متن", "word " * 20
            )
        )
        self.assertTrue(
            translation.translation_is_usable(
                "جس کسی کی نماز نہیں ہوئی", "Whoever has not performed the prayer"
            )
        )

    def test_primary_exception_uses_fallback(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(service, "_translate_with_fallback"))
        service.translator = FakeTranslator(error=RuntimeError("primary failed"))
        service.fallback_translator = FakeTranslator(
            result=translation_result("Whoever has not performed the prayer")
        )

        result = service._translate_with_fallback(["اردو متن"], ["ur"], 1)

        self.assertEqual(["Whoever has not performed the prayer"], result.texts)
        self.assertEqual(1, service.fallback_translator.calls)

    def test_unusable_primary_output_uses_fallback(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(service, "_translate_with_fallback"))
        service.translator = FakeTranslator(result=translation_result("اردو متن"))
        service.fallback_translator = FakeTranslator(
            result=translation_result("Urdu text")
        )

        result = service._translate_with_fallback(["اردو متن"], ["ur"], 1)

        self.assertEqual(["Urdu text"], result.texts)
        self.assertEqual(1, service.fallback_translator.calls)

    def test_valid_primary_output_does_not_use_fallback(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(service, "_translate_with_fallback"))
        service.translator = FakeTranslator(result=translation_result("Urdu text"))
        service.fallback_translator = FakeTranslator(
            result=translation_result("fallback")
        )

        result = service._translate_with_fallback(["اردو متن"], ["ur"], 1)

        self.assertEqual(["Urdu text"], result.texts)
        self.assertEqual(0, service.fallback_translator.calls)

    def test_fallback_failure_is_wrapped_as_translation_output_error(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(service, "_translate_with_fallback"))
        service.translator = FakeTranslator(error=RuntimeError("primary failed"))
        service.fallback_translator = FakeTranslator(
            error=RuntimeError("fallback failed")
        )

        with self.assertRaises(Exception) as caught:
            service._translate_with_fallback(["اردو متن"], ["ur"], 1)

        self.assertEqual("TranslationOutputError", type(caught.exception).__name__)


class InferenceTimeoutTests(unittest.TestCase):
    def test_generation_kwargs_include_soft_time_limit(self):
        translator = translation.Translator.__new__(translation.Translator)
        translator.cfg = types.SimpleNamespace(family="indictrans2")

        kwargs = translator._generate_kwargs(source_tokens=10)

        self.assertIn("max_time", kwargs)
        self.assertGreater(kwargs["max_time"], 0)

    def test_watchdog_fires_after_hard_deadline(self):
        spec = importlib.util.find_spec("src.inference_watchdog")
        self.assertIsNotNone(spec)
        watchdog_module = importlib.import_module("src.inference_watchdog")
        fired = threading.Event()

        watchdog = watchdog_module.InferenceWatchdog(0.02, abort=fired.set)
        watchdog.arm()

        self.assertTrue(fired.wait(0.5))

    def test_cancelled_watchdog_does_not_fire(self):
        spec = importlib.util.find_spec("src.inference_watchdog")
        self.assertIsNotNone(spec)
        watchdog_module = importlib.import_module("src.inference_watchdog")
        fired = threading.Event()

        watchdog = watchdog_module.InferenceWatchdog(0.02, abort=fired.set)
        watchdog.arm()
        watchdog.cancel()
        time.sleep(0.05)

        self.assertFalse(fired.is_set())

    def test_cancelled_watchdog_ignores_late_timer_callback(self):
        watchdog_module = importlib.import_module("src.inference_watchdog")
        fired = threading.Event()
        watchdog = watchdog_module.InferenceWatchdog(10, abort=fired.set)

        watchdog.arm()
        watchdog.cancel()
        self.assertTrue(hasattr(watchdog, "_fire"))
        watchdog._fire()

        self.assertFalse(fired.is_set())

    def test_guarded_model_call_returns_operation_result(self):
        watchdog_module = importlib.import_module("src.inference_watchdog")
        self.assertTrue(hasattr(watchdog_module, "guarded_model_call"))

        result = watchdog_module.guarded_model_call(lambda: "completed", 1)

        self.assertEqual("completed", result)


class HealthStatusTests(unittest.TestCase):
    def test_stage_status_reports_urdu_fallback_and_deadlines(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        service.device = types.SimpleNamespace(type="cuda")
        service.detector = types.SimpleNamespace(backend="lingua")
        service.roman_detector = types.SimpleNamespace(ftr=object(), bert=None)
        service.transliterator = types.SimpleNamespace(models={"ur": object()})
        service.translator = types.SimpleNamespace(
            cfg=types.SimpleNamespace(display_name="IndicTrans2"),
            preprocessing_mode="IndicTransToolkit",
            urdu_preprocessing_ready=True,
        )
        service.fallback_translator = types.SimpleNamespace(
            cfg=types.SimpleNamespace(display_name="NLLB")
        )
        service.classifier = types.SimpleNamespace(
            cfg=types.SimpleNamespace(display_name="Cardiff")
        )

        status = service.stage_status()

        self.assertIn("urdu_preprocessing_ready", status)
        self.assertIn("fallback_translator", status)
        self.assertIn("generation_max_time_s", status)
        self.assertIn("inference_hard_timeout_s", status)
        self.assertTrue(status["urdu_preprocessing_ready"])
        self.assertEqual("NLLB", status["fallback_translator"])
        self.assertGreater(status["generation_max_time_s"], 0)
        self.assertGreater(status["inference_hard_timeout_s"], 0)


class ApiFailureContractTests(unittest.TestCase):
    def test_translation_failure_returns_503(self):
        import api_server
        from fastapi import HTTPException

        original_pipeline = api_server._pipeline
        original_run = api_server._run_pipeline
        api_server._pipeline = types.SimpleNamespace()

        def fail(_texts, _request_id):
            raise translation.TranslationOutputError("both translators failed")

        api_server._run_pipeline = fail
        try:
            with self.assertRaises(Exception) as caught:
                api_server.analyze(api_server.AnalyzeRequest(texts=["اردو متن"]))
        finally:
            api_server._pipeline = original_pipeline
            api_server._run_pipeline = original_run

        self.assertIsInstance(caught.exception, HTTPException)
        self.assertEqual(503, caught.exception.status_code)


if __name__ == "__main__":
    unittest.main()
