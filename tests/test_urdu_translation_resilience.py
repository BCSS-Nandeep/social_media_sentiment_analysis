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

        result = service.transliterate("Teen Din meh 6th Sat hai,", "ur")

        self.assertEqual("تین دن میں 6th Sat ہے,", result)
        self.assertEqual(["__ur__ h a i"], model.prepared)

    def test_indicxlit_input_includes_language_token_and_lowercase_chars(self):
        self.assertTrue(hasattr(transliteration, "_prepare_word"))

        prepared = transliteration._prepare_word("Namaz", "ur")

        self.assertEqual("__ur__ n a m a z", prepared)

    def test_bert_can_override_confident_ftr_english_for_roman_urdu(self):
        detector = lid_roman.RomanLanguageDetector.__new__(
            lid_roman.RomanLanguageDetector
        )
        detector._ftr_topk = lambda _text: [("eng_Latn", 0.999), ("urd_Latn", 0.001)]
        detector._bert_predict = lambda _text: "urd_Latn"

        language = detector.detect(
            "Jis kissi ki bhi Taraweeh ki Namaz Nahi hui hai"
        )

        self.assertEqual("ur", language)


class PipelineOutcomeTests(unittest.TestCase):
    @staticmethod
    def _result(text):
        return pipeline.PipelineResult(
            post_text=text,
            language="en",
            english_text=text,
            was_translated=False,
            was_transliterated=False,
            sentiment="Neutral",
            confidence=0.8,
            translation_time_ms=0.0,
            sentiment_time_ms=1.0,
            total_time_ms=1.0,
        )

    def test_batch_failure_isolates_only_the_failed_post(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(pipeline, "PipelineFailure"))
        self.assertTrue(hasattr(service, "predict_batch_outcomes"))
        service._lock = threading.Lock()

        def predict_locked(texts, *_args):
            if len(texts) > 1:
                raise translation.TranslationOutputError("batch failed")
            if texts[0] == "bad":
                raise translation.TranslationOutputError("post failed")
            return [self._result(texts[0])]

        service._predict_batch_locked = predict_locked
        service._failure_context = lambda _text: ("ur", True)

        outcomes = service.predict_batch_outcomes(["first", "bad", "last"])

        self.assertIsInstance(outcomes[0], pipeline.PipelineResult)
        self.assertIsInstance(outcomes[1], pipeline.PipelineFailure)
        self.assertIsInstance(outcomes[2], pipeline.PipelineResult)
        self.assertEqual("bad", outcomes[1].post_text)
        self.assertEqual("ur", outcomes[1].language)
        self.assertTrue(outcomes[1].was_transliterated)

    def test_compatible_predict_batch_raises_when_failure_remains(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        self.assertTrue(hasattr(pipeline, "PipelineFailure"))
        service.predict_batch_outcomes = lambda *_args, **_kwargs: [
            pipeline.PipelineFailure(
                post_text="bad",
                language="ur",
                was_transliterated=True,
                deterministic_time_ms=2.0,
                reason="post failed",
            )
        ]

        with self.assertRaises(translation.TranslationOutputError):
            service.predict_batch(["bad"])

    def test_invalid_deterministic_result_becomes_failure(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        service._lock = threading.Lock()
        invalid = self._result("bad")
        invalid.sentiment = "Mixed"
        service._predict_batch_locked = lambda *_args, **_kwargs: [invalid]

        outcomes = service.predict_batch_outcomes(["bad"])

        self.assertIsInstance(outcomes[0], pipeline.PipelineFailure)

    def test_nonrecoverable_failure_propagates_without_item_retries(self):
        service = pipeline.SentimentPipeline.__new__(pipeline.SentimentPipeline)
        service._lock = threading.Lock()
        calls = []

        def fail(texts, *_args):
            calls.append(list(texts))
            raise MemoryError("model allocation failed")

        service._predict_batch_locked = fail

        with self.assertRaises(MemoryError):
            service.predict_batch_outcomes(["first", "last"])

        self.assertEqual([["first", "last"]], calls)


class LlmFallbackTests(unittest.TestCase):
    class FakeProvider:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error
            self.calls = []

        def generate(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            if self.error:
                raise self.error
            return self.response

        def describe(self):
            return {"provider": "vllm", "model": "test-model"}

        def health(self):
            return True

        def close(self):
            return None

    @staticmethod
    def _failure():
        return pipeline.PipelineFailure(
            post_text="Jis kisi ki namaz nahi hui",
            language="ur",
            was_transliterated=True,
            deterministic_time_ms=12.5,
            reason="TranslationOutputError",
            translation_time_ms=7.5,
        )

    def _module(self):
        spec = importlib.util.find_spec("src.pipeline_fallback")
        self.assertIsNotNone(spec)
        return importlib.import_module("src.pipeline_fallback")

    def test_valid_provider_response_matches_pipeline_result_schema(self):
        module = self._module()
        provider = self.FakeProvider(
            {
                "english_text": "Whoever has not performed the prayer",
                "sentiment": "Neutral",
                "confidence": 0.82,
            }
        )
        fallback = module.LlmPipelineFallback(
            provider=provider, timeout_s=5
        )

        result = fallback.resolve(self._failure())

        expected_keys = set(PipelineOutcomeTests._result("x").to_dict())
        self.assertEqual(expected_keys, set(result.to_dict()))
        self.assertEqual("Neutral", result.sentiment)
        self.assertEqual(0.82, result.confidence)
        self.assertEqual("ur", result.language)
        self.assertTrue(result.was_translated)
        self.assertTrue(result.was_transliterated)
        self.assertGreaterEqual(result.translation_time_ms, 7.5)
        self.assertEqual(1, len(provider.calls))

    def test_invalid_provider_shapes_are_rejected(self):
        module = self._module()
        invalid = [
            {"english_text": "English", "sentiment": "Neutral"},
            {
                "english_text": "English",
                "sentiment": "Neutral",
                "confidence": 0.5,
                "extra": "not allowed",
            },
            {
                "english_text": "English",
                "sentiment": "Mixed",
                "confidence": 0.5,
            },
            {
                "english_text": "English",
                "sentiment": "Positive",
                "confidence": True,
            },
            {
                "english_text": "English",
                "sentiment": "Negative",
                "confidence": float("nan"),
            },
            {
                "english_text": "English",
                "sentiment": "Negative",
                "confidence": 1.1,
            },
            {
                "english_text": "Jis kisi ki namaz nahi hui",
                "sentiment": "Neutral",
                "confidence": 0.5,
            },
        ]

        for response in invalid:
            with self.subTest(response=response):
                fallback = module.LlmPipelineFallback(
                    provider=self.FakeProvider(response), timeout_s=5
                )
                with self.assertRaises(module.PipelineFallbackError):
                    fallback.resolve(self._failure())

    def test_provider_error_is_normalized(self):
        module = self._module()
        fallback = module.LlmPipelineFallback(
            provider=self.FakeProvider(error=TimeoutError("offline")),
            timeout_s=5,
        )

        with self.assertRaises(module.PipelineFallbackError):
            fallback.resolve(self._failure())

    def test_pipeline_english_validator_rejects_roman_urdu_output(self):
        module = self._module()
        provider = self.FakeProvider(
            {
                "english_text": "Jis ki niyaz nahi thi",
                "sentiment": "Neutral",
                "confidence": 0.8,
            }
        )
        fallback = module.LlmPipelineFallback(
            provider=provider,
            timeout_s=5,
            english_validator=lambda _text: False,
        )

        with self.assertRaises(module.PipelineFallbackError):
            fallback.resolve(self._failure())


class FallbackQueueTests(unittest.TestCase):
    def _module(self):
        spec = importlib.util.find_spec("src.fallback_queue")
        self.assertIsNotNone(spec)
        return importlib.import_module("src.fallback_queue")

    def test_queue_is_fifo_and_rejects_work_at_capacity(self):
        module = self._module()
        work_queue = module.BoundedWorkQueue(
            workers=1,
            capacity=1,
            thread_name_prefix="fallback-test",
        )
        started = threading.Event()
        release = threading.Event()
        order = []

        def first():
            started.set()
            release.wait(1)
            order.append("first")
            return "first"

        try:
            first_future = work_queue.submit(first, timeout_s=0)
            self.assertTrue(started.wait(0.5))
            second_future = work_queue.submit(
                lambda: order.append("second") or "second",
                timeout_s=0,
            )

            with self.assertRaises(module.FallbackQueueFull):
                work_queue.submit(lambda: "third", timeout_s=0)

            stats = work_queue.stats()
            self.assertEqual(1, stats["active"])
            self.assertEqual(1, stats["queued"])
            self.assertEqual(1, stats["rejected"])

            release.set()
            self.assertEqual("first", first_future.result(timeout=1))
            self.assertEqual("second", second_future.result(timeout=1))
            self.assertEqual(["first", "second"], order)
        finally:
            release.set()
            work_queue.shutdown()

    def test_batch_admission_is_all_or_none(self):
        module = self._module()
        work_queue = module.BoundedWorkQueue(
            workers=1,
            capacity=1,
            thread_name_prefix="fallback-test",
        )
        try:
            with self.assertRaises(module.FallbackQueueFull):
                work_queue.submit_many(
                    [
                        (lambda: "first", (), {}),
                        (lambda: "second", (), {}),
                    ],
                    timeout_s=0,
                )
            self.assertEqual(0, work_queue.stats()["accepted"])
            self.assertEqual(2, work_queue.stats()["rejected"])
        finally:
            work_queue.shutdown()

    def test_cancelling_pending_work_releases_capacity_immediately(self):
        module = self._module()
        work_queue = module.BoundedWorkQueue(
            workers=1,
            capacity=1,
            thread_name_prefix="fallback-test",
        )
        started = threading.Event()
        release = threading.Event()
        try:
            active = work_queue.submit(
                lambda: started.set() or release.wait(1),
                timeout_s=0,
            )
            self.assertTrue(started.wait(0.5))
            pending = work_queue.submit(lambda: "pending", timeout_s=0)
            self.assertTrue(pending.cancel())

            replacement = work_queue.submit(
                lambda: "replacement", timeout_s=0
            )
            self.assertEqual(1, work_queue.stats()["cancelled"])
            release.set()
            active.result(timeout=1)
            self.assertEqual("replacement", replacement.result(timeout=1))
        finally:
            release.set()
            work_queue.shutdown()

    def test_shutdown_rejects_a_submit_waiting_for_capacity(self):
        module = self._module()
        work_queue = module.BoundedWorkQueue(
            workers=1,
            capacity=1,
            thread_name_prefix="fallback-test",
        )
        started = threading.Event()
        release = threading.Event()
        submit_error = []
        shutdown_thread = None
        submit_thread = None
        try:
            work_queue.submit(
                lambda: started.set() or release.wait(1),
                timeout_s=0,
            )
            self.assertTrue(started.wait(0.5))
            work_queue.submit(lambda: "pending", timeout_s=0)

            def wait_to_submit():
                try:
                    work_queue.submit(lambda: "late", timeout_s=1)
                except Exception as exc:
                    submit_error.append(exc)

            submit_thread = threading.Thread(target=wait_to_submit)
            submit_thread.start()
            time.sleep(0.02)
            shutdown_thread = threading.Thread(target=work_queue.shutdown)
            shutdown_thread.start()
            submit_thread.join(timeout=0.5)

            self.assertFalse(submit_thread.is_alive())
            self.assertIsInstance(submit_error[0], RuntimeError)
        finally:
            release.set()
            if submit_thread is not None:
                submit_thread.join(timeout=1)
            if shutdown_thread is not None:
                shutdown_thread.join(timeout=1)
            work_queue.shutdown()


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
        self.assertFalse(
            translation.translation_is_usable(
                "చాలా worst గా ఉంది", "It has been going on"
            )
        )
        self.assertTrue(
            translation.translation_is_usable(
                "చాలా worst గా ఉంది", "It is the worst"
            )
        )
        self.assertTrue(
            translation.translation_is_usable(
                "చాలా worst గా ఉంది", "It's very bad"
            )
        )
        self.assertTrue(
            translation.translation_is_usable(
                "good ga chesaru", "well done"
            )
        )
        self.assertFalse(
            translation.translation_is_usable(
                "చాలా worst గా ఉంది", "It's very good"
            )
        )

    def test_long_posts_split_on_sentence_boundaries(self):
        chunks = translation.split_translation_chunks(
            "First sentence. Second sentence! Third one?"
        )
        self.assertEqual(
            ["First sentence.", "Second sentence!", "Third one?"],
            chunks,
        )
        self.assertEqual(
            ["short undi"],
            translation.split_translation_chunks("short undi"),
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

    def test_api_health_reports_pipeline_fallback_readiness(self):
        import api_server

        class FakeFallback:
            def describe(self):
                return {
                    "enabled": True,
                    "provider": "vllm",
                    "model": "test-model",
                    "timeout_s": 60,
                    "max_workers": 2,
                }

            def health(self):
                return True

        original_pipeline = api_server._pipeline
        original_analyzer = api_server._analyzer
        original_fallback = api_server._pipeline_fallback
        original_queue = api_server._pipeline_fallback_queue
        api_server._pipeline = types.SimpleNamespace(
            device=types.SimpleNamespace(type="cuda"),
            stage_status=lambda: {},
        )
        api_server._analyzer = None
        api_server._pipeline_fallback = FakeFallback()
        api_server._pipeline_fallback_queue = types.SimpleNamespace(
            stats=lambda: {
                "workers": 2,
                "capacity": 64,
                "queued": 0,
                "active": 0,
                "accepted": 3,
                "rejected": 0,
                "completed": 3,
            }
        )
        try:
            payload = api_server.health()
        finally:
            api_server._pipeline = original_pipeline
            api_server._analyzer = original_analyzer
            api_server._pipeline_fallback = original_fallback
            api_server._pipeline_fallback_queue = original_queue

        self.assertTrue(payload["pipeline_fallback"]["available"])
        self.assertTrue(payload["pipeline_fallback"]["reachable"])
        self.assertEqual("vllm", payload["pipeline_fallback"]["provider"])
        self.assertEqual(2, payload["pipeline_fallback"]["max_workers"])
        self.assertEqual(64, payload["pipeline_fallback"]["queue"]["capacity"])


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


class ApiFallbackOrchestrationTests(unittest.TestCase):
    def test_only_failures_use_fallback_after_inference_lock_release(self):
        import api_server

        first = PipelineOutcomeTests._result("first")
        last = PipelineOutcomeTests._result("last")
        failure = pipeline.PipelineFailure(
            post_text="bad",
            language="ur",
            was_transliterated=True,
            deterministic_time_ms=3.0,
            reason="TranslationOutputError",
            translation_time_ms=1.0,
        )

        class FakePipeline:
            def predict_batch_outcomes(self, *_args, **_kwargs):
                self.lock_was_held = api_server._inference_lock.locked()
                return [first, failure, last]

        class FakeFallback:
            def __init__(self):
                self.calls = []

            def resolve(self, item):
                self.calls.append(
                    (item.post_text, api_server._inference_lock.locked())
                )
                return pipeline.PipelineResult(
                    post_text=item.post_text,
                    language=item.language,
                    english_text="translated",
                    was_translated=True,
                    was_transliterated=item.was_transliterated,
                    sentiment="Neutral",
                    confidence=0.7,
                    translation_time_ms=4.0,
                    sentiment_time_ms=0.0,
                    total_time_ms=4.0,
                )

        original_pipeline = api_server._pipeline
        original_fallback = getattr(api_server, "_pipeline_fallback", None)
        fake_pipeline = FakePipeline()
        fake_fallback = FakeFallback()
        api_server._pipeline = fake_pipeline
        api_server._pipeline_fallback = fake_fallback
        try:
            self.assertTrue(hasattr(api_server, "_run_pipeline_with_fallback"))
            results = api_server._run_pipeline_with_fallback(
                ["first", "bad", "last"], request_id=99
            )
        finally:
            api_server._pipeline = original_pipeline
            api_server._pipeline_fallback = original_fallback

        self.assertTrue(fake_pipeline.lock_was_held)
        self.assertEqual([("bad", False)], fake_fallback.calls)
        self.assertEqual(["first", "bad", "last"], [r.post_text for r in results])
        self.assertEqual(4.0, results[1].translation_time_ms)

    def test_full_fallback_queue_returns_pipeline_fallback_error(self):
        import api_server

        queue_module = importlib.import_module("src.fallback_queue")
        failure = pipeline.PipelineFailure(
            post_text="bad",
            language="ur",
            was_transliterated=True,
            deterministic_time_ms=3.0,
            reason="TranslationOutputError",
            translation_time_ms=1.0,
        )

        class FullQueue:
            def submit_many(self, *_args, **_kwargs):
                raise queue_module.FallbackQueueFull("queue full")

            def stats(self):
                return {"queued": 1, "capacity": 1, "rejected": 1}

        original_fallback = api_server._pipeline_fallback
        self.assertTrue(hasattr(api_server, "_pipeline_fallback_queue"))
        original_queue = api_server._pipeline_fallback_queue
        api_server._pipeline_fallback = types.SimpleNamespace(
            resolve=lambda _failure: None
        )
        api_server._pipeline_fallback_queue = FullQueue()
        try:
            with self.assertRaises(api_server.PipelineFallbackError):
                api_server._resolve_pipeline_outcomes([failure], request_id=100)
        finally:
            api_server._pipeline_fallback = original_fallback
            api_server._pipeline_fallback_queue = original_queue


if __name__ == "__main__":
    unittest.main()
