"""Stage 3 — intelligence layer over the deterministic pipeline's output.

    post -> [ language detection -> romanized-Indic LID -> IndicXlit ->
              IndicTrans2 -> Cardiff RoBERTa ]      <- src/pipeline.py, unchanged
         -> intent | category | risk | reasoning | summary | recommended action

This module NEVER re-does anything the pipeline already decided. Language,
translation, transliteration, sentiment and sentiment confidence arrive here as
*trusted facts* and are passed to the model as given. The model's job is the
part deterministic classifiers cannot do: reading the translated text in context
and producing an analyst-facing judgement.

Callers may optionally supply a *policy pack* (category allowlist + definitions)
so the prompt and Ollama JSON schema are built for that taxonomy. Omitting the
pack reproduces the built-in CATEGORY_LABELS behaviour exactly.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import config
from src.ollama_gate import (
    OllamaCircuitOpen,
    OllamaGateFull,
    get_ollama_gate,
)
from src.script_detection import LATIN_RANGE, UNICODE_RANGES

logger = logging.getLogger("benchmark.intelligence")

_ALLOWED_INTENTS = frozenset(config.INTENT_LABELS) | {config.UNKNOWN_LABEL}
_ALLOWED_ACTIONS = frozenset(config.ACTION_LABELS)
_ALLOWED_EVIDENCE = frozenset(("high", "medium", "low"))
_INTENT_MODES = frozenset(("enum", "free"))


# --------------------------------------------------------------------------- #
# Policy pack — caller-supplied category taxonomy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyCategory:
    id: str
    definition: str = ""
    severity: str = ""
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyPack:
    """Category allowlist used to build the system prompt and response schema."""

    name: str
    categories: tuple[PolicyCategory, ...]
    unknown_label: str
    version: str = ""
    fingerprint: str = ""

    def category_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.categories)

    def allowed_categories(self) -> frozenset[str]:
        return frozenset(self.category_ids())


def _canonical_pack_payload(
    categories: tuple[PolicyCategory, ...], unknown_label: str, intent_mode: str
) -> str:
    payload = {
        "unknown_label": unknown_label,
        "intent_mode": intent_mode,
        "categories": [
            {
                "id": c.id,
                "definition": c.definition,
                "severity": c.severity,
                "keywords": list(c.keywords),
            }
            for c in categories
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint_pack(
    categories: tuple[PolicyCategory, ...], unknown_label: str, intent_mode: str = "enum"
) -> str:
    digest = hashlib.sha256(
        _canonical_pack_payload(categories, unknown_label, intent_mode).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def build_default_policy_pack() -> PolicyPack:
    """Built-in taxonomy — identical behaviour to the pre-policy-pack service."""
    categories = tuple(PolicyCategory(id=label) for label in config.CATEGORY_LABELS)
    # Unknown is always selectable for insufficient evidence when using the default pack.
    if config.UNKNOWN_LABEL not in {c.id for c in categories}:
        categories = (*categories, PolicyCategory(id=config.UNKNOWN_LABEL))
    return PolicyPack(
        name="default",
        categories=categories,
        unknown_label=config.UNKNOWN_LABEL,
        version="builtin",
        fingerprint=fingerprint_pack(categories, config.UNKNOWN_LABEL, "enum"),
    )


DEFAULT_POLICY_PACK: PolicyPack = build_default_policy_pack()


def parse_policy_pack(raw: dict | None) -> PolicyPack:
    """Validate a caller-supplied pack dict, or return the default pack.

    Raises ValueError with a caller-safe message on structural problems (API maps to 422).
    """
    if not raw:
        return DEFAULT_POLICY_PACK

    categories_raw = raw.get("categories")
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ValueError("policy_pack.categories must be a non-empty list")
    if len(categories_raw) > config.INTELLIGENCE_MAX_POLICY_CATEGORIES:
        raise ValueError(
            f"policy_pack.categories has {len(categories_raw)} entries "
            f"(limit {config.INTELLIGENCE_MAX_POLICY_CATEGORIES})"
        )

    seen: set[str] = set()
    categories: list[PolicyCategory] = []
    for i, entry in enumerate(categories_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"policy_pack.categories[{i}] must be an object")
        cat_id = str(entry.get("id") or "").strip()
        if not cat_id:
            raise ValueError(f"policy_pack.categories[{i}].id is required")
        if len(cat_id) > 96:
            raise ValueError(f"policy_pack.categories[{i}].id exceeds 96 characters")
        if cat_id in seen:
            raise ValueError(f"policy_pack.categories contains duplicate id {cat_id!r}")
        seen.add(cat_id)
        definition = str(entry.get("definition") or "")
        if len(definition) > 500:
            definition = definition[:500]
            logger.info("Truncated definition for category %s to 500 chars", cat_id)
        keywords_raw = entry.get("keywords") or []
        if not isinstance(keywords_raw, list):
            raise ValueError(f"policy_pack.categories[{i}].keywords must be a list")
        keywords = tuple(str(k).strip() for k in keywords_raw if str(k).strip())
        categories.append(
            PolicyCategory(
                id=cat_id,
                definition=definition,
                severity=str(entry.get("severity") or "").strip(),
                keywords=keywords,
            )
        )

    unknown = str(raw.get("unknown_label") or "").strip()
    cat_tuple = tuple(categories)
    if unknown:
        if unknown not in seen:
            raise ValueError(
                f"policy_pack.unknown_label {unknown!r} must be one of the category ids"
            )
    else:
        # Append the service Unknown label when the caller did not nominate one.
        unknown = config.UNKNOWN_LABEL
        if unknown not in seen:
            cat_tuple = (*cat_tuple, PolicyCategory(id=unknown))

    name = str(raw.get("name") or "caller").strip() or "caller"
    version = str(raw.get("version") or "").strip()
    fp = fingerprint_pack(cat_tuple, unknown, "enum")
    # Prefer a caller fingerprint only when it matches; otherwise use the canonical one.
    caller_fp = str(raw.get("fingerprint") or "").strip()
    if caller_fp and caller_fp != fp:
        logger.info(
            "policy_pack fingerprint mismatch (caller=%s computed=%s) — using computed",
            caller_fp, fp,
        )
    return PolicyPack(
        name=name,
        categories=cat_tuple,
        unknown_label=unknown,
        version=version,
        fingerprint=fp,
    )


# --------------------------------------------------------------------------- #
# Dynamic system prompt + response schema
# --------------------------------------------------------------------------- #
def _category_block(pack: PolicyPack) -> str:
    lines: list[str] = []
    for cat in pack.categories:
        if cat.definition:
            lines.append(f"  {cat.id} — {cat.definition}")
        else:
            lines.append(f"  {cat.id}")
        if cat.id == pack.unknown_label:
            lines[-1] += (
                " Use this when nothing else applies AND when the evidence is "
                "insufficient to choose."
            )
    return "\n".join(lines)


def _intent_block(intent_mode: str) -> str:
    labels = ", ".join(config.INTENT_LABELS)
    if intent_mode == "free":
        return (
            f"intent — a short free-form phrase of 2 to {config.INTELLIGENCE_INTENT_MAX_WORDS} "
            "words naming what the author is trying to achieve "
            "(e.g. 'mobilize a protest march', 'warn residents of flooding'). "
            "Do not use a full sentence.\n"
            f"intent_label — additionally classify that intent as exactly one of: "
            f'{labels}, or "{config.UNKNOWN_LABEL}"'
        )
    return (
        f"intent — exactly one of:\n  {labels}, or \"{config.UNKNOWN_LABEL}\""
    )


def _response_template(intent_mode: str) -> str:
    if intent_mode == "free":
        return (
            '{"category": "...", "intent": "...", "intent_label": "...", '
            '"risk_score": 0, "reasoning": "...", "summary": "...", '
            '"recommended_action": "...", "evidence_confidence": "..."}'
        )
    return (
        '{"category": "...", "intent": "...", "risk_score": 0, "reasoning": "...", '
        '"summary": "...", "recommended_action": "...", "evidence_confidence": "..."}'
    )


def _insufficient_block(intent_mode: str, unknown: str) -> str:
    if intent_mode == "free":
        return (
            f'  category           -> "{unknown}"\n'
            f'  intent             -> "{unknown}" (or a short uncertain phrase)\n'
            f'  intent_label       -> "{config.UNKNOWN_LABEL}"\n'
            f"  risk_score         -> 0\n"
            f"  reasoning          -> explain exactly what is missing\n"
            f'  recommended_action -> "Human Review"\n'
            f'  evidence_confidence-> "low"'
        )
    return (
        f'  category           -> "{unknown}"\n'
        f'  intent             -> "{config.UNKNOWN_LABEL}"\n'
        f"  risk_score         -> 0\n"
        f"  reasoning          -> explain exactly what is missing\n"
        f'  recommended_action -> "Human Review"\n'
        f'  evidence_confidence-> "low"'
    )


@lru_cache(maxsize=64)
def build_system_prompt(fingerprint: str, intent_mode: str, pack_json: str) -> str:
    """Memoized on fingerprint + intent_mode. pack_json is the canonical payload."""
    del fingerprint  # used only as cache key companion; pack_json carries data
    payload = json.loads(pack_json)
    categories = tuple(
        PolicyCategory(
            id=c["id"],
            definition=c.get("definition", ""),
            severity=c.get("severity", ""),
            keywords=tuple(c.get("keywords") or ()),
        )
        for c in payload["categories"]
    )
    pack = PolicyPack(
        name="cached",
        categories=categories,
        unknown_label=payload["unknown_label"],
        fingerprint="",
    )
    unknown = pack.unknown_label
    return f"""\
You are an intelligence analysis engine. You operate as the SECOND stage of a \
pipeline. A deterministic NLP pipeline has already processed the post and its \
results are given to you as established facts.

TRUSTED INPUTS — treat these as ground truth, never recompute them:
  language, english_text, sentiment, confidence, was_translated, was_transliterated

ABSOLUTE RULES:
1. Never translate the input text. The translation is already provided.
2. Never detect, guess, or comment on the language. It is already provided.
3. Never perform sentiment analysis.
4. Never modify, override, second-guess or "correct" the supplied sentiment.
5. Never ignore the structured inputs. Every judgement must use them.
6. Base all reasoning strictly on: the original text, the English translation, \
the supplied sentiment, the confidence score, and the supplied signals.
7. If the evidence is insufficient, say so explicitly and request further \
evidence. Do not fill gaps with assumptions.
8. Never invent facts, events, people, organizations, locations or \
relationships that are not present in the provided input.
9. Stay objective and evidence-based. Do not assert criminal intent and do not \
make factual claims beyond the supplied content. Describe what the text says, \
not what it might imply about the real world.
10. Return ONLY the JSON object. No prose, no Markdown, no code fences, no \
commentary before or after.
11. If `matched_keywords` are provided in the input, evaluate whether each keyword \
is used in a genuinely threatening or malicious context. Do NOT assume a keyword \
indicates risk just because it is present. Provide your assessment in the \
`keyword_context` field.

TRUSTED INPUTS — treat these as ground truth, never recompute them:
  language, english_text, sentiment, confidence, was_translated, was_transliterated

ABSOLUTE RULES:
1. Never translate the input text. The translation is already provided.
2. Never detect, guess, or comment on the language. It is already provided.
3. Never perform sentiment analysis.
4. Never modify, override, second-guess or "correct" the supplied sentiment.
5. Never ignore the structured inputs. Every judgement must use them.
6. Base all reasoning strictly on: the original text, the English translation, \
the supplied sentiment, the confidence score, and the supplied signals.
7. If the evidence is insufficient, say so explicitly and request further \
evidence. Do not fill gaps with assumptions.
8. Never invent facts, events, people, organizations, locations or \
relationships that are not present in the provided input.
9. Stay objective and evidence-based. Do not assert criminal intent and do not \
make factual claims beyond the supplied content. Describe what the text says, \
not what it might imply about the real world.
10. Return ONLY the JSON object. No prose, no Markdown, no code fences, no \
commentary before or after.

YOUR TASKS — intent, category, contextual risk, reasoning, summary, action.

{_intent_block(intent_mode)}

category — exactly one of the following. Choose the single best fit.
{_category_block(pack)}

risk_score — integer 0-100, a CONTEXTUAL risk assessment weighing:
  likelihood of public disorder, communal sensitivity, violence indicators,
  misinformation potential, criminal relevance, intelligence significance.
  Do NOT derive this from sentiment alone. A strongly negative post can be
  low risk (ordinary complaint) and a neutral or positive post can be high
  risk (calm, organized mobilization). Sentiment is one input among several.

reasoning — why this category and this risk score, citing the specific content
  that drove it. Name the signals you relied on. State any uncertainty.

summary — one or two sentences, written for an investigator reading a
  dashboard. Factual, no speculation.

recommended_action — exactly one of:
  {", ".join(config.ACTION_LABELS)}
  with the justification carried in `reasoning`.

evidence_confidence — "high", "medium" or "low": your confidence in the above
  given the amount and clarity of the evidence available.

EDGE CASES — handle these explicitly rather than guessing:
- Very short, ambiguous or incomplete posts: prefer "{unknown}",
  lower the risk score, set evidence_confidence "low", recommend "Human Review".
- Sarcasm or irony: mark uncertain rather than overconfident. Say so in reasoning.
- Heavy code-mixing (Hinglish/Tenglish/etc.): translation may be imperfect;
  reduce confidence accordingly instead of over-reading the English.
- Translation or transliteration failure signalled in the input: the English
  text may be unreliable. Say so and prefer "Human Review".
- Quoted or forwarded content: assess the content, but note in reasoning that
  the author may not be the originator.
- Spam, advertisement or duplicate content: usually low risk, "Ignore" or
  "Monitor" — unless the content itself is a fraud or recruitment lure.
- OCR-noisy text: if the text is too garbled to read reliably, say so and
  recommend "Human Review" rather than guessing at meaning.
- Truncated posts (signalled as `truncated`): note that you saw part of the post.
- Low sentiment confidence: treat the sentiment as weaker evidence, not as fact
  to be overturned — you still must not recompute it.
- Conflicting signals, e.g. Positive sentiment alongside threatening language:
  do NOT resolve this by changing the sentiment. Report the conflict in
  reasoning, weigh the CONTENT for risk, and raise the recommended action.

INSUFFICIENT EVIDENCE — when you cannot determine a field confidently:
{_insufficient_block(intent_mode, unknown)}

Return only this JSON object:
{_response_template(intent_mode)}
"""


def _prompt_for(pack: PolicyPack, intent_mode: str, has_keywords: bool = False) -> str:
    pack_json = _canonical_pack_payload(pack.categories, pack.unknown_label, intent_mode)
    fp = fingerprint_pack(pack.categories, pack.unknown_label, intent_mode)
    prompt = build_system_prompt(fp, intent_mode, pack_json)
    return prompt


@lru_cache(maxsize=64)
def build_response_schema(fingerprint: str, intent_mode: str, category_enum_json: str, has_keywords: bool = False) -> dict:
    del fingerprint
    category_enum = json.loads(category_enum_json)
    properties: dict = {
        "category": {"type": "string", "enum": category_enum},
        "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
        "recommended_action": {"type": "string", "enum": sorted(_ALLOWED_ACTIONS)},
        "evidence_confidence": {"type": "string", "enum": sorted(_ALLOWED_EVIDENCE)},
    }
    required = [
        "category", "intent", "risk_score", "reasoning", "summary",
        "recommended_action", "evidence_confidence",
    ]
    if intent_mode == "free":
        properties["intent"] = {"type": "string"}
        properties["intent_label"] = {
            "type": "string",
            "enum": sorted(_ALLOWED_INTENTS),
        }
        required.append("intent_label")
    else:
        properties["intent"] = {"type": "string", "enum": sorted(_ALLOWED_INTENTS)}
        
    properties["keyword_context"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "matched": {"type": "boolean"},
                "contextually_relevant": {"type": "boolean"},
                "usage": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["keyword", "matched", "contextually_relevant", "usage", "reason"]
        }
    }
    
    if has_keywords:
        required.append("keyword_context")
        
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _schema_for(pack: PolicyPack, intent_mode: str, has_keywords: bool = False) -> dict:
    fp = fingerprint_pack(pack.categories, pack.unknown_label, intent_mode)
    return build_response_schema(
        fp, intent_mode, json.dumps(sorted(pack.category_ids())), has_keywords
    )


# Backward-compatible module-level defaults (no-pack path).
SYSTEM_PROMPT: str = _prompt_for(DEFAULT_POLICY_PACK, "enum")
RESPONSE_SCHEMA: dict = _schema_for(DEFAULT_POLICY_PACK, "enum")
_ALLOWED_CATEGORIES = DEFAULT_POLICY_PACK.allowed_categories()


@dataclass
class IntelligenceResult:
    """One post's intelligence record. Stored alongside the pipeline output."""

    category: str
    intent: str
    risk_score: int
    reasoning: str
    summary: str
    recommended_action: str
    evidence_confidence: str          # high | medium | low
    signals: list[str] = field(default_factory=list)  # deterministic edge-case flags
    source: str = "provider"          # provider | triage | error — how this was produced
    model: str = ""
    latency_ms: float = 0.0
    intent_label: str = ""            # enum label; equals intent in enum mode
    policy_pack_fingerprint: str = ""
    schema_enforced: bool = True
    keyword_context: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _insufficient(
    reasoning: str,
    signals: list[str],
    source: str,
    *,
    pack: PolicyPack | None = None,
    intent_mode: str = "enum",
    action: str = "Human Review",
) -> IntelligenceResult:
    """The mandated shape for 'not enough evidence' — used by both triage and
    every failure path, so an unreadable post and an unreachable Ollama produce
    records a consumer can treat identically."""
    pack = pack or DEFAULT_POLICY_PACK
    unknown = pack.unknown_label
    intent = unknown if intent_mode == "free" else (
        config.UNKNOWN_LABEL if config.UNKNOWN_LABEL in _ALLOWED_INTENTS else unknown
    )
    intent_label = config.UNKNOWN_LABEL if intent_mode == "free" else intent
    return IntelligenceResult(
        category=unknown,
        intent=intent,
        intent_label=intent_label,
        risk_score=0,
        reasoning=reasoning,
        summary="No intelligence assessment could be produced for this post.",
        recommended_action=action,
        evidence_confidence="low",
        signals=signals,
        source=source,
        policy_pack_fingerprint=fingerprint_pack(
            pack.categories, pack.unknown_label, intent_mode
        ),
        schema_enforced=True,
        keyword_context=[],
    )


# --------------------------------------------------------------------------- #
# Deterministic edge-case detection
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MEDIA_RE = re.compile(
    r"^\W*(?:\[|<|\()?\s*(?:image|photo|video|media|gif|sticker|audio|voice|"
    r"document|file)\s*(?:omitted|attached|unavailable)?\s*(?:\]|>|\))?\W*$",
    re.IGNORECASE,
)
_FORWARD_RE = re.compile(r"^\s*(?:>|RT\s+@|fwd:|forwarded(?:\s+message)?:)", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")
_SPAM_RE = re.compile(
    r"\b(?:buy\s+now|click\s+here|limited\s+offer|whatsapp\s+me|dm\s+me|"
    r"earn\s+\d|100%\s+guarantee|subscribe|promo\s*code)\b",
    re.IGNORECASE,
)


def _script_mix(text: str) -> float:
    """Fraction of alphabetic characters in the minority script (Latin vs Indic)."""
    latin = indic = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if LATIN_RANGE[0] <= cp <= LATIN_RANGE[1]:
            latin += 1
            continue
        for lo, hi in UNICODE_RANGES.values():
            if lo <= cp <= hi:
                indic += 1
                break
    total = latin + indic
    if total == 0:
        return 0.0
    return min(latin, indic) / total


def derive_signals(result: dict) -> list[str]:
    """Cheap, deterministic edge-case flags attached to the prompt."""
    original = str(result.get("post_text") or "")
    english = str(result.get("english_text") or "")
    language = str(result.get("language") or "")
    signals: list[str] = []

    words = english.split()
    if len(words) <= config.INTELLIGENCE_SHORT_WORDS:
        signals.append("very_short")
    if len(original) > config.INTELLIGENCE_MAX_TEXT_CHARS:
        signals.append("truncated")
    if float(result.get("confidence") or 0.0) < config.INTELLIGENCE_LOW_CONFIDENCE:
        signals.append("low_sentiment_confidence")
    if _script_mix(original) >= config.INTELLIGENCE_CODEMIX_RATIO:
        signals.append("code_mixed")
    if result.get("was_transliterated"):
        signals.append("romanized_indic_transliterated")
    if result.get("fallback_used"):
        signals.append("ollama_pipeline_fallback")
    if result.get("translation_truncated"):
        signals.append("translation_truncated")
    if result.get("sentiment_truncated"):
        signals.append("sentiment_truncated")
    if language not in ("en", "") and not result.get("was_translated"):
        signals.append("translation_unavailable")
    if language == "unknown":
        signals.append("language_undetermined")
    if _FORWARD_RE.search(original):
        signals.append("quoted_or_forwarded")
    if _SPAM_RE.search(original) or len(_HASHTAG_RE.findall(original)) >= 8:
        signals.append("possible_spam")
    if len(_URL_RE.findall(original)) >= 3:
        signals.append("link_heavy")
    if len(words) >= 6 and sum(1 for w in words if len(w) == 1) / len(words) > 0.4:
        signals.append("possible_ocr_noise")
    return signals


def triage(
    result: dict,
    signals: list[str],
    *,
    pack: PolicyPack | None = None,
    intent_mode: str = "enum",
) -> IntelligenceResult | None:
    """Resolve posts with no analyzable content without calling the model."""
    pack = pack or DEFAULT_POLICY_PACK
    original = str(result.get("post_text") or "")
    english = str(result.get("english_text") or "")

    if not original.strip():
        return _insufficient(
            "The post is empty or contains only whitespace; there is no content to assess.",
            signals, "triage", pack=pack, intent_mode=intent_mode, action="Ignore",
        )
    if _MEDIA_RE.match(original.strip()):
        return _insufficient(
            "The post is a media placeholder with no accompanying text. The "
            "attached media itself was not available for assessment.",
            signals, "triage", pack=pack, intent_mode=intent_mode,
        )
    stripped = _URL_RE.sub(" ", original)
    if not any(ch.isalnum() for ch in stripped):
        reason = (
            "The post contains no textual content once links are removed "
            "(emoji, punctuation or URL only), so intent, category and "
            "contextual risk cannot be assessed from text."
        )
        return _insufficient(
            reason, signals, "triage", pack=pack, intent_mode=intent_mode, action="Ignore",
        )
    if not english.strip():
        return _insufficient(
            "The pipeline produced no English text for this post, so no "
            "assessment can be made from the translation.",
            signals, "triage", pack=pack, intent_mode=intent_mode,
        )
    return None


def build_payload(result: dict, signals: list[str], matched_keywords: list[dict] | None = None) -> dict:
    """The structured user message — pipeline facts, never bare text."""
    def clip(text: str) -> str:
        limit = config.INTELLIGENCE_MAX_TEXT_CHARS
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.7)]
        tail = text[-int(limit * 0.3) :]
        return f"{head}\n...[truncated]...\n{tail}"

    payload = {
        "original_text": clip(str(result.get("post_text") or "")),
        "language": result.get("language"),
        "english_text": clip(str(result.get("english_text") or "")),
        "sentiment": result.get("sentiment"),
        "confidence": result.get("confidence"),
        "was_translated": bool(result.get("was_translated")),
        "was_transliterated": bool(result.get("was_transliterated")),
        "translation_backend": result.get("translation_backend") or "unknown",
        "fallback_used": bool(result.get("fallback_used")),
        "fallback_reason": str(result.get("fallback_reason") or ""),
        "transliterated_text": clip(str(result.get("transliterated_text") or "")),
        "signals": signals,
    }
    if matched_keywords:
        payload["matched_keywords"] = matched_keywords
    return payload


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class IntelligenceProvider(ABC):
    """Extension point. Register new backends in :data:`PROVIDERS`."""

    name: str = "provider"

    @abstractmethod
    def generate(
        self,
        payload: dict,
        *,
        system_prompt: str,
        schema: dict,
        timeout_s: float | None = None,
    ) -> dict:
        """Return the model's parsed JSON object, or raise."""

    @abstractmethod
    def describe(self) -> dict:
        """Static config, surfaced by /health."""

    def schema_enforced(self) -> bool:
        return True

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None


class OllamaProvider(IntelligenceProvider):
    """Ollama /api/chat, constrained to the response schema where supported."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        model: str = config.OLLAMA_MODEL,
        timeout_s: float = config.OLLAMA_TIMEOUT_S,
        retries: int = config.OLLAMA_RETRIES,
        use_schema: bool = config.OLLAMA_JSON_SCHEMA,
    ) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self._use_schema = use_schema
        self._session = requests.Session()
        self._lock = threading.Lock()

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "json_schema": self._use_schema,
            "num_predict": config.OLLAMA_NUM_PREDICT,
            "default_policy_pack_fingerprint": DEFAULT_POLICY_PACK.fingerprint,
        }

    def schema_enforced(self) -> bool:
        return self._use_schema

    def health(self) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Ollama health check failed for %s: %s", self.base_url, exc)
            return False

    def _body(
        self,
        payload: dict,
        use_schema: bool,
        system_prompt: str,
        schema: dict,
    ) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": config.OLLAMA_NUM_PREDICT,
                "seed": config.SEED,
            },
        }
        body["format"] = schema if use_schema else "json"
        return body

    def generate(
        self,
        payload: dict,
        *,
        system_prompt: str,
        schema: dict,
        timeout_s: float | None = None,
    ) -> dict:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        last_error: Exception | None = None
        gate = get_ollama_gate()
        with gate.slot():
            for attempt in range(self.retries + 1):
                use_schema = self._use_schema
                try:
                    response = self._session.post(
                        f"{self.base_url}/api/chat",
                        json=self._body(payload, use_schema, system_prompt, schema),
                        timeout=timeout,
                    )
                    if use_schema and response.status_code == 400:
                        with self._lock:
                            if self._use_schema:
                                logger.warning(
                                    "Ollama at %s rejected a JSON schema (400) — falling "
                                    "back to format=\"json\" for the rest of this process.",
                                    self.base_url,
                                )
                                self._use_schema = False
                        response = self._session.post(
                            f"{self.base_url}/api/chat",
                            json=self._body(payload, False, system_prompt, schema),
                            timeout=timeout,
                        )
                    response.raise_for_status()
                    content = response.json().get("message", {}).get("content", "")
                    return _parse_json_object(content)
                except Exception as exc:
                    last_error = exc
                    if attempt < self.retries:
                        logger.warning(
                            "Ollama attempt %d/%d failed (%s) — retrying.",
                            attempt + 1, self.retries + 1, exc,
                        )
            raise RuntimeError(
                f"Ollama request failed after {self.retries + 1} attempt(s): {last_error}"
            )

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


PROVIDERS: dict[str, type[IntelligenceProvider]] = {
    OllamaProvider.name: OllamaProvider,
}


def _parse_json_object(content: str) -> dict:
    """Decode the model's reply, tolerating a stray code fence or prose."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in model reply: {content[:200]!r}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Model reply was not a JSON object: {content[:200]!r}")
    return parsed


def _clamp_free_intent(value: str) -> str:
    words = value.split()
    max_words = max(1, config.INTELLIGENCE_INTENT_MAX_WORDS)
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return value.strip()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class IntelligenceAnalyzer:
    """Turns pipeline results into intelligence records.

    Stateless per call and safe to share across threads. Batches fan out across
    ``OLLAMA_CONCURRENCY`` workers; identical posts within a batch are analyzed
    once and the record reused.
    """

    def __init__(self, provider: IntelligenceProvider | None = None) -> None:
        if provider is None:
            name = config.INTELLIGENCE_PROVIDER
            if name not in PROVIDERS:
                raise ValueError(
                    f"Unknown intelligence provider {name!r}; "
                    f"available: {sorted(PROVIDERS)}"
                )
            provider = PROVIDERS[name]()
        self.provider = provider
        logger.info("Intelligence layer ready: %s", self.provider.describe())

    def describe(self) -> dict:
        return self.provider.describe()

    def health(self) -> bool:
        return self.provider.health()

    def analyze_one(
        self,
        result: dict,
        *,
        pack: PolicyPack | None = None,
        intent_mode: str = "enum",
        timeout_s: float | None = None,
        matched_keywords: list[dict] | None = None,
    ) -> IntelligenceResult:
        """Never raises — every failure becomes an 'insufficient evidence' record."""
        pack = pack or DEFAULT_POLICY_PACK
        if intent_mode not in _INTENT_MODES:
            intent_mode = "enum"

        signals = derive_signals(result)
        short_circuit = triage(result, signals, pack=pack, intent_mode=intent_mode)
        if short_circuit is not None:
            return short_circuit

        system_prompt = _prompt_for(pack, intent_mode, has_keywords=bool(matched_keywords))
        schema = _schema_for(pack, intent_mode, has_keywords=bool(matched_keywords))
        started = time.perf_counter()
        try:
            raw = self.provider.generate(
                build_payload(result, signals, matched_keywords=matched_keywords),
                system_prompt=system_prompt,
                schema=schema,
                timeout_s=timeout_s,
            )
        except (OllamaGateFull, OllamaCircuitOpen):
            raise
        except Exception as exc:
            logger.error("Intelligence provider failed: %s", exc)
            record = _insufficient(
                f"The intelligence provider could not be reached or returned an "
                f"unusable response ({type(exc).__name__}). The deterministic "
                f"sentiment result is unaffected and remains valid.",
                signals, "error", pack=pack, intent_mode=intent_mode,
            )
            record.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
            record.model = getattr(self.provider, "model", "")
            record.schema_enforced = self.provider.schema_enforced()
            return record

        record = self._validate(raw, signals, pack=pack, intent_mode=intent_mode)
        record.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        record.model = getattr(self.provider, "model", "")
        record.schema_enforced = self.provider.schema_enforced()
        return record

    def analyze_batch(
        self,
        results: list[dict],
        *,
        pack: PolicyPack | None = None,
        intent_mode: str = "enum",
        timeout_s: float | None = None,
        matched_keywords: list[list[dict]] | None = None,
    ) -> list[IntelligenceResult]:
        """Order-preserving. Duplicate posts share one provider call."""
        if not results:
            return []

        pack = pack or DEFAULT_POLICY_PACK
        if intent_mode not in _INTENT_MODES:
            intent_mode = "enum"
        pack_fp = fingerprint_pack(pack.categories, pack.unknown_label, intent_mode)

        first_index: dict[tuple, int] = {}
        todo: list[int] = []
        reuse: dict[int, int] = {}
        for i, result in enumerate(results):
            key = (
                result.get("post_text"),
                result.get("english_text"),
                result.get("sentiment"),
                pack_fp,
                intent_mode,
            )
            if key in first_index:
                reuse[i] = first_index[key]
                continue
            first_index[key] = i
            todo.append(i)

        workers = max(
            1,
            min(config.OLLAMA_CONCURRENCY, get_ollama_gate().size, len(todo)),
        )
        started = time.perf_counter()
        computed: dict[int, IntelligenceResult] = {}

        def _run(j: int) -> IntelligenceResult:
            mks = matched_keywords[j] if matched_keywords and j < len(matched_keywords) else None
            return self.analyze_one(
                results[j], pack=pack, intent_mode=intent_mode, timeout_s=timeout_s, matched_keywords=mks
            )

        if workers == 1:
            for i in todo:
                computed[i] = _run(i)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, record in zip(todo, pool.map(_run, todo)):
                    computed[i] = record

        out: list[IntelligenceResult] = []
        for i in range(len(results)):
            if i in reuse:
                source = computed[reuse[i]]
                clone = IntelligenceResult(**source.to_dict())
                if "duplicate_post" not in clone.signals:
                    clone.signals = [*clone.signals, "duplicate_post"]
                out.append(clone)
            else:
                out.append(computed[i])

        errors = sum(1 for r in out if r.source == "error")
        logger.info(
            "Intelligence: %d post(s) in %.0f ms — %d analyzed, %d deduplicated, "
            "%d resolved without a model call, %d failed (pack=%s mode=%s)",
            len(results), (time.perf_counter() - started) * 1000.0,
            len(todo), len(reuse),
            sum(1 for r in out if r.source == "triage"), errors,
            pack.name, intent_mode,
        )
        return out

    def _validate(
        self,
        raw: dict,
        signals: list[str],
        *,
        pack: PolicyPack,
        intent_mode: str,
    ) -> IntelligenceResult:
        """Clamp the model's reply onto the contract."""
        allowed_categories = pack.allowed_categories()

        def pick(key: str, allowed: frozenset[str], default: str) -> str:
            value = str(raw.get(key, "") or "").strip()
            if value in allowed:
                return value
            for candidate in allowed:
                if candidate.lower() == value.lower():
                    return candidate
            if value:
                logger.warning("Model returned %s=%r, outside the taxonomy.", key, value)
            return default

        category = pick("category", allowed_categories, pack.unknown_label)
        action = pick("recommended_action", _ALLOWED_ACTIONS, "Human Review")
        evidence = pick("evidence_confidence", _ALLOWED_EVIDENCE, "low")

        if intent_mode == "free":
            intent = _clamp_free_intent(str(raw.get("intent", "") or "").strip())
            if not intent:
                intent = pack.unknown_label
            intent_label = pick("intent_label", _ALLOWED_INTENTS, config.UNKNOWN_LABEL)
        else:
            intent = pick("intent", _ALLOWED_INTENTS, config.UNKNOWN_LABEL)
            intent_label = intent

        try:
            risk = int(round(float(raw.get("risk_score", 0))))
        except (TypeError, ValueError):
            logger.warning("Model returned a non-numeric risk_score %r.", raw.get("risk_score"))
            risk = 0
        risk = max(0, min(100, risk))

        reasoning = str(raw.get("reasoning", "") or "").strip()
        summary = str(raw.get("summary", "") or "").strip()
        if not reasoning:
            reasoning = "The model returned no reasoning for this assessment."
            action = "Human Review"
        if not summary:
            summary = "No summary was produced for this post."

        keyword_context = []
        if isinstance(raw.get("keyword_context"), list):
            keyword_context = raw["keyword_context"]

        return IntelligenceResult(
            category=category,
            intent=intent,
            intent_label=intent_label,
            risk_score=risk,
            reasoning=reasoning,
            summary=summary,
            recommended_action=action,
            evidence_confidence=evidence,
            signals=signals,
            source="provider",
            policy_pack_fingerprint=fingerprint_pack(
                pack.categories, pack.unknown_label, intent_mode
            ),
            keyword_context=keyword_context,
        )

    def close(self) -> None:
        self.provider.close()
