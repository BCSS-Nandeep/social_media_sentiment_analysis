"""Tenant-aware request / stance response contract for api_server.py.

Calls the FastAPI route functions directly (no ASGI, no lifespan, no real
model load) with stubbed pipeline/analyzer globals — this repo's models are
multi-GB downloads that must not be a precondition for these tests. Mirrors
the "no live vLLM required" convention already used by
verify_intelligence_contract.py.

Covers:
  * tenant_name is optional and defaults to None (backward compatibility)
  * /analyze ignores tenant_name entirely — same response keys as before
  * /analyze/intelligence forwards tenant_name to the intelligence layer,
    and omits it (None) when the caller sends none
  * invalid tenant_name (too long / control characters) is rejected with 422
  * the response carries stance / stance_confidence
  * the existing 429 (gate full) contract, including Retry-After, survives
    unchanged with tenant_name present
"""
from __future__ import annotations

import unittest

import api_server
from src.llm_gate import LlmGateFull
from src.pipeline import PipelineResult


def _pipeline_result(text: str) -> PipelineResult:
    return PipelineResult(
        post_text=text,
        language="en",
        english_text=text,
        was_translated=False,
        was_transliterated=False,
        sentiment="Positive",
        confidence=0.9,
        translation_time_ms=0.0,
        sentiment_time_ms=1.0,
        total_time_ms=1.0,
    )


class StubPipeline:
    """Minimal stand-in for SentimentPipeline.predict_batch_outcomes."""

    def __init__(self, outcomes=None, raises: Exception | None = None):
        self._outcomes = outcomes
        self._raises = raises
        self.device = type("D", (), {"type": "cpu"})()

    def predict_batch_outcomes(self, texts, *, batch_size, translation_batch_size, request_id):
        if self._raises is not None:
            raise self._raises
        return self._outcomes or [_pipeline_result(t) for t in texts]

    def stage_status(self):
        return {}


class StubAnalyzer:
    """Records the kwargs analyze_batch was called with."""

    def __init__(self, records=None, raises: Exception | None = None):
        self._records = records
        self._raises = raises
        self.last_kwargs: dict | None = None

    def analyze_batch(self, payloads, **kwargs):
        self.last_kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        if self._records is not None:
            return self._records
        return [_StubRecord() for _ in payloads]


class _StubRecord:
    def to_dict(self) -> dict:
        return {
            "category": "Other", "intent": "Information", "intent_label": "Information",
            "risk_score": 0, "reasoning": "r", "summary": "s",
            "recommended_action": "Ignore", "evidence_confidence": "low",
            "signals": [], "source": "provider", "model": "stub", "latency_ms": 1.0,
            "policy_pack_fingerprint": "sha256:stub", "schema_enforced": True,
            "keyword_context": [], "stance": "Neutral", "stance_confidence": 0.5,
        }


class ApiTenantAndStanceTests(unittest.TestCase):
    def setUp(self):
        self._orig_pipeline = api_server._pipeline
        self._orig_analyzer = api_server._analyzer

    def tearDown(self):
        api_server._pipeline = self._orig_pipeline
        api_server._analyzer = self._orig_analyzer

    # -- request schema -----------------------------------------------------
    def test_tenant_name_defaults_to_none(self):
        req = api_server.AnalyzeRequest(texts=["hello"])
        self.assertIsNone(req.tenant_name)

    def test_legacy_request_without_tenant_name_still_parses(self):
        req = api_server.AnalyzeRequest.model_validate({"texts": ["hello"]})
        self.assertIsNone(req.tenant_name)

    # -- /analyze ignores tenant_name ---------------------------------------
    def test_analyze_ignores_tenant_name(self):
        api_server._pipeline = StubPipeline()
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="Tenant 1")
        out = api_server.analyze(req)
        result = out["results"][0]
        self.assertNotIn("tenant_name", result)
        self.assertEqual(result["sentiment"], "Positive")

    # -- /analyze/intelligence forwards tenant_name --------------------------
    def test_intelligence_forwards_tenant_name(self):
        api_server._pipeline = StubPipeline()
        analyzer = StubAnalyzer()
        api_server._analyzer = analyzer
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="Tenant 1")
        out = api_server.analyze_intelligence(req)
        self.assertEqual(analyzer.last_kwargs["tenant_name"], "Tenant 1")
        intel = out["results"][0]["intelligence"]
        self.assertEqual(intel["stance"], "Neutral")
        self.assertEqual(intel["stance_confidence"], 0.5)

    def test_intelligence_tenant_name_omitted_is_none(self):
        api_server._pipeline = StubPipeline()
        analyzer = StubAnalyzer()
        api_server._analyzer = analyzer
        req = api_server.AnalyzeRequest(texts=["hello"])
        api_server.analyze_intelligence(req)
        self.assertIsNone(analyzer.last_kwargs["tenant_name"])

    # -- tenant_name validation edge cases -----------------------------------
    def test_tenant_name_too_long_is_rejected(self):
        api_server._pipeline = StubPipeline()
        api_server._analyzer = StubAnalyzer()
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="x" * 500)
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server.analyze_intelligence(req)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_tenant_name_control_chars_rejected(self):
        api_server._pipeline = StubPipeline()
        api_server._analyzer = StubAnalyzer()
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="bad\x00name")
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server.analyze_intelligence(req)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_empty_tenant_name_normalizes_to_none(self):
        api_server._pipeline = StubPipeline()
        analyzer = StubAnalyzer()
        api_server._analyzer = analyzer
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="   ")
        api_server.analyze_intelligence(req)
        self.assertIsNone(analyzer.last_kwargs["tenant_name"])

    def test_missing_stance_defaults_via_record_shape(self):
        # A record shape from an older analyzer/version without stance keys
        # must not crash the endpoint — dict merge just omits the field.
        class LegacyRecord:
            def to_dict(self):
                return {"category": "Other", "risk_score": 0}

        api_server._pipeline = StubPipeline()
        api_server._analyzer = StubAnalyzer(records=[LegacyRecord()])
        req = api_server.AnalyzeRequest(texts=["hello"])
        out = api_server.analyze_intelligence(req)
        self.assertNotIn("stance", out["results"][0]["intelligence"])

    # -- 429 / backpressure contract preserved -------------------------------
    def test_429_contract_preserved_on_analyze_with_tenant_name(self):
        api_server._pipeline = StubPipeline(raises=LlmGateFull(retry_after_s=7))
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="Tenant 1")
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server.analyze(req)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.headers["Retry-After"], "7")

    def test_429_contract_preserved_on_intelligence_gate(self):
        api_server._pipeline = StubPipeline()
        api_server._analyzer = StubAnalyzer(raises=LlmGateFull(retry_after_s=9))
        req = api_server.AnalyzeRequest(texts=["hello"], tenant_name="Tenant 1")
        with self.assertRaises(api_server.HTTPException) as ctx:
            api_server.analyze_intelligence(req)
        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.headers["Retry-After"], "9")


if __name__ == "__main__":
    unittest.main()
