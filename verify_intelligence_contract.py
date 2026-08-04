#!/usr/bin/env python3
"""Contract checks for the intelligence policy-pack API (no live Ollama required).

Asserts:
  * schema is built from the supplied pack
  * every returned category ∈ pack
  * unknown_label is honoured on triage / provider failure
  * free intent is clamped to INTELLIGENCE_INTENT_MAX_WORDS
  * intent_label ∈ enum in free mode
  * provider failure yields source="error"
  * triage yields source="triage"
  * exactly one provider generate() call per unique post

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
        self.model = "stub-model"

    def generate(self, payload, *, system_prompt, schema, timeout_s=None):
        self.calls += 1
        self.last_schema = schema
        self.last_prompt = system_prompt
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
