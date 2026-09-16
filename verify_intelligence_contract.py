#!/usr/bin/env python3
"""Contract checks for the intelligence policy-pack API (no live vLLM required).

Asserts:
  * schema is built from the supplied pack
  * every returned category ∈ pack
  * unknown_label is honoured on triage / provider failure
  * free intent is clamped to INTELLIGENCE_INTENT_MAX_WORDS
  * intent_label ∈ enum in free mode
  * provider failure yields source="error"
  * triage yields source="triage"
  * exactly one provider generate() call per unique post
  * stance is a fixed enum, independent of the category policy pack
  * stance is not derived from sentiment (positive != Support, negative != Oppose)
  * malformed stance / stance_confidence from the model is clamped, never raises
  * triage/error paths carry a well-formed stance ("Unclear", confidence 0.0)
  * tenant_name reaches the provider payload as context and never as a field
    the model is asked to classify

Run from the service root:
    python verify_intelligence_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.intelligence import (
    DEFAULT_POLICY_PACK,
    IntelligenceAnalyzer,
    IntelligenceProvider,
    PolicyCategory,
    PolicyPack,
    build_payload,
    _schema_for,
    fingerprint_pack,
    parse_policy_pack,
)


class StubProvider(IntelligenceProvider):
    """Returns canned JSON and counts generate() calls."""

    name = "stub"

    def __init__(self, canned: dict | None = None, fail: bool = False) -> None:
        self.canned = canned or {}
        self.fail = fail
        self.calls = 0
        self.last_schema: dict | None = None
        self.last_prompt: str = ""
        self.last_payload: dict | None = None
        self.model = "stub-model"

    def generate(self, payload, *, system_prompt, schema, timeout_s=None):
        self.calls += 1
        self.last_schema = schema
        self.last_prompt = system_prompt
        self.last_payload = payload
        if self.fail:
            raise RuntimeError("stub provider deliberately failing")
        return dict(self.canned)

    def describe(self) -> dict:
        return {"provider": self.name, "model": self.model}

    def schema_enforced(self) -> bool:
        return True


def _pipeline_payload(**overrides) -> dict:
    base = {
        "post_text": "The government announced a new welfare scheme today.",
        "language": "en",
        "english_text": "The government announced a new welfare scheme today.",
        "sentiment": "Positive",
        "confidence": 0.92,
        "was_translated": False,
        "was_transliterated": False,
    }
    base.update(overrides)
    return base


def _sockeye_pack() -> PolicyPack:
    categories = (
        PolicyCategory(
            id="Communal_Violence",
            definition="Content inciting violence between religious communities.",
        ),
        PolicyCategory(id="Hate_Speech", definition="Content attacking a protected group."),
        PolicyCategory(id="Normal", definition="Harmless or neutral content."),
    )
    return PolicyPack(
        name="sockeye-test",
        categories=categories,
        unknown_label="Normal",
        fingerprint=fingerprint_pack(categories, "Normal", "enum"),
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_default_pack_schema_matches_builtin() -> None:
    schema = _schema_for(DEFAULT_POLICY_PACK, "enum")
    enums = set(schema["properties"]["category"]["enum"])
    _assert(config.UNKNOWN_LABEL in enums, "default schema must include Unknown")
    for label in config.CATEGORY_LABELS:
        _assert(label in enums, f"default schema missing {label}")


def test_pack_schema_uses_caller_ids() -> None:
    pack = _sockeye_pack()
    schema = _schema_for(pack, "free")
    enums = schema["properties"]["category"]["enum"]
    _assert(enums == sorted(pack.category_ids()), f"schema enum mismatch: {enums}")
    _assert("intent_label" in schema["properties"], "free mode requires intent_label")
    _assert(schema["properties"]["intent"]["type"] == "string", "free intent is plain string")


def test_parse_rejects_bad_unknown() -> None:
    try:
        parse_policy_pack({
            "categories": [{"id": "A"}],
            "unknown_label": "Missing",
        })
    except ValueError as exc:
        _assert("unknown_label" in str(exc), str(exc))
    else:
        raise AssertionError("expected ValueError for unknown_label not in ids")


def test_triage_uses_unknown_label() -> None:
    provider = StubProvider(fail=True)
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    record = analyzer.analyze_one(
        _pipeline_payload(post_text="   ", english_text=""),
        pack=pack,
        intent_mode="free",
    )
    _assert(record.source == "triage", f"expected triage, got {record.source}")
    _assert(record.category == "Normal", f"expected Normal, got {record.category}")
    _assert(provider.calls == 0, "triage must not call the provider")


def test_provider_failure_is_error_source() -> None:
    provider = StubProvider(fail=True)
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    record = analyzer.analyze_one(_pipeline_payload(), pack=pack, intent_mode="free")
    _assert(record.source == "error", f"expected error, got {record.source}")
    _assert(record.category == "Normal", f"expected Normal category on error")
    _assert(record.risk_score == 0, "error path must force risk_score 0")
    _assert(provider.calls >= 1, "provider must have been called")


def test_category_clamped_to_pack() -> None:
    provider = StubProvider(canned={
        "category": "NOT_IN_PACK",
        "intent": "spread communal fear online",
        "intent_label": "Propaganda",
        "risk_score": 55,
        "reasoning": "test",
        "summary": "test summary",
        "recommended_action": "Monitor",
        "evidence_confidence": "medium",
    })
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    record = analyzer.analyze_one(_pipeline_payload(), pack=pack, intent_mode="free")
    _assert(record.category == "Normal", f"out-of-pack category should fall back to unknown_label, got {record.category}")
    _assert(record.intent_label == "Propaganda", record.intent_label)
    _assert(record.category in pack.allowed_categories(), record.category)


def test_free_intent_word_clamp() -> None:
    long_intent = "one two three four five six seven eight nine ten eleven"
    provider = StubProvider(canned={
        "category": "Hate_Speech",
        "intent": long_intent,
        "intent_label": "Threat",
        "risk_score": 80,
        "reasoning": "test",
        "summary": "test summary",
        "recommended_action": "Escalate",
        "evidence_confidence": "high",
    })
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    record = analyzer.analyze_one(_pipeline_payload(), pack=pack, intent_mode="free")
    words = record.intent.split()
    _assert(
        len(words) <= config.INTELLIGENCE_INTENT_MAX_WORDS,
        f"intent has {len(words)} words: {record.intent!r}",
    )
    _assert(record.intent_label == "Threat", record.intent_label)
    _assert("Communal_Violence" in provider.last_prompt or "Hate_Speech" in provider.last_prompt,
            "system prompt should list pack categories")


def test_batch_dedupe_one_call() -> None:
    provider = StubProvider(canned={
        "category": "Normal",
        "intent": "share news",
        "intent_label": "Information",
        "risk_score": 5,
        "reasoning": "benign",
        "summary": "benign post",
        "recommended_action": "Ignore",
        "evidence_confidence": "high",
    })
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    payload = _pipeline_payload()
    records = analyzer.analyze_batch(
        [payload, dict(payload), dict(payload)],
        pack=pack,
        intent_mode="free",
    )
    _assert(provider.calls == 1, f"expected 1 provider call, got {provider.calls}")
    _assert(len(records) == 3, len(records))
    _assert(all(r.category == "Normal" for r in records), "all clones share category")
    _assert("duplicate_post" in records[1].signals, records[1].signals)


def _canned(**overrides) -> dict:
    base = {
        "category": "Normal",
        "intent": "share news",
        "intent_label": "Information",
        "risk_score": 5,
        "reasoning": "benign",
        "summary": "benign post",
        "recommended_action": "Ignore",
        "evidence_confidence": "high",
        "stance": "Neutral",
        "stance_confidence": 0.5,
    }
    base.update(overrides)
    return base


def test_schema_has_stance_fields() -> None:
    schema = _schema_for(DEFAULT_POLICY_PACK, "enum")
    _assert("stance" in schema["properties"], "schema missing stance")
    _assert("stance_confidence" in schema["properties"], "schema missing stance_confidence")
    _assert(set(schema["properties"]["stance"]["enum"]) == set(config.STANCE_LABELS),
            schema["properties"]["stance"]["enum"])
    _assert("stance" in schema["required"] and "stance_confidence" in schema["required"],
            "stance fields must be required")


def test_stance_not_derived_from_sentiment() -> None:
    """A model may report Oppose stance on a post the pipeline scored Positive
    sentiment — the two axes are independent and neither may override the other."""
    provider = StubProvider(canned=_canned(stance="Oppose", stance_confidence=0.8))
    analyzer = IntelligenceAnalyzer(provider=provider)
    pack = _sockeye_pack()
    record = analyzer.analyze_one(
        _pipeline_payload(sentiment="Positive"), pack=pack, intent_mode="free",
    )
    _assert(record.stance == "Oppose", f"expected Oppose, got {record.stance}")
    _assert(record.stance_confidence == 0.8, record.stance_confidence)

    provider2 = StubProvider(canned=_canned(stance="Support", stance_confidence=0.7))
    record2 = IntelligenceAnalyzer(provider=provider2).analyze_one(
        _pipeline_payload(sentiment="Negative"), pack=pack, intent_mode="free",
    )
    _assert(record2.stance == "Support", f"expected Support, got {record2.stance}")


def test_stance_out_of_enum_falls_back_to_unclear() -> None:
    provider = StubProvider(canned=_canned(stance="Definitely Yes"))
    record = IntelligenceAnalyzer(provider=provider).analyze_one(
        _pipeline_payload(), pack=_sockeye_pack(), intent_mode="free",
    )
    _assert(record.stance == config.UNKNOWN_STANCE, record.stance)


def test_stance_confidence_clamped_and_never_nan() -> None:
    cases = [
        (5, 1.0), (-3, 0.0), (float("nan"), 0.0), (float("inf"), 0.0),
        ("not-a-number", 0.0), (None, 0.0), (0.42, 0.42),
    ]
    for raw, expected in cases:
        provider = StubProvider(canned=_canned(stance_confidence=raw))
        record = IntelligenceAnalyzer(provider=provider).analyze_one(
            _pipeline_payload(), pack=_sockeye_pack(), intent_mode="free",
        )
        _assert(
            record.stance_confidence == expected,
            f"stance_confidence({raw!r}) -> {record.stance_confidence}, expected {expected}",
        )
        _assert(record.stance_confidence == record.stance_confidence, "NaN leaked through")


def test_triage_and_error_stance_is_unclear() -> None:
    triage_record = IntelligenceAnalyzer(provider=StubProvider(fail=True)).analyze_one(
        _pipeline_payload(post_text="   ", english_text=""), pack=_sockeye_pack(),
    )
    _assert(triage_record.stance == config.UNKNOWN_STANCE, triage_record.stance)
    _assert(triage_record.stance_confidence == 0.0, triage_record.stance_confidence)

    error_record = IntelligenceAnalyzer(provider=StubProvider(fail=True)).analyze_one(
        _pipeline_payload(), pack=_sockeye_pack(),
    )
    _assert(error_record.stance == config.UNKNOWN_STANCE, error_record.stance)
    _assert(error_record.stance_confidence == 0.0, error_record.stance_confidence)


def test_tenant_name_reaches_payload_as_context_only() -> None:
    provider = StubProvider(canned=_canned())
    analyzer = IntelligenceAnalyzer(provider=provider)
    analyzer.analyze_one(
        _pipeline_payload(), pack=_sockeye_pack(), intent_mode="free",
        tenant_name="Tenant 1",
    )
    _assert(provider.last_payload is not None, "provider was not called")
    _assert(provider.last_payload.get("tenant_name") == "Tenant 1",
            provider.last_payload)
    schema = _schema_for(_sockeye_pack(), "free")
    _assert("tenant_name" not in schema["properties"],
            "tenant_name must never be a field the model classifies")

    provider2 = StubProvider(canned=_canned())
    IntelligenceAnalyzer(provider=provider2).analyze_one(
        _pipeline_payload(), pack=_sockeye_pack(), intent_mode="free",
    )
    _assert("tenant_name" not in (provider2.last_payload or {}),
            "omitted tenant_name must not appear in the payload")


def test_build_payload_backward_compatible_without_tenant_name() -> None:
    payload = build_payload(_pipeline_payload(), [])
    _assert("tenant_name" not in payload, "legacy call site must be unaffected")


def main() -> int:
    tests = [
        test_default_pack_schema_matches_builtin,
        test_pack_schema_uses_caller_ids,
        test_parse_rejects_bad_unknown,
        test_triage_uses_unknown_label,
        test_provider_failure_is_error_source,
        test_category_clamped_to_pack,
        test_free_intent_word_clamp,
        test_batch_dedupe_one_call,
        test_schema_has_stance_fields,
        test_stance_not_derived_from_sentiment,
        test_stance_out_of_enum_falls_back_to_unclear,
        test_stance_confidence_clamped_and_never_nan,
        test_triage_and_error_stance_is_unclear,
        test_tenant_name_reaches_payload_as_context_only,
        test_build_payload_backward_compatible_without_tenant_name,
    ]
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print()
    if failed:
        print(f"{failed}/{len(tests)} failed")
        return 1
    print(f"All {len(tests)} contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
