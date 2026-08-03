"""Stage 3 — intelligence layer over the deterministic pipeline's output.

    post -> [ language detection -> romanized-Indic LID -> IndicXlit ->
              IndicTrans2 -> Cardiff RoBERTa ]      <- src/pipeline.py, unchanged
         -> intent | category | risk | reasoning | summary | recommended action

This module NEVER re-does anything the pipeline already decided. Language,
translation, transliteration, sentiment and sentiment confidence arrive here as
*trusted facts* and are passed to the model as given. The model's job is the
part deterministic classifiers cannot do: reading the translated text in context
and producing an analyst-facing judgement.

Design notes
------------
* **Additive.** Nothing in src/pipeline.py or the /analyze contract changes.
  This stage consumes a PipelineResult (or its dict) and returns a separate
  IntelligenceResult; callers store both.
* **Pluggable.** :class:`IntelligenceProvider` is the extension point and
  providers are looked up by name in :data:`PROVIDERS`, so another backend can
  be added later without touching the sentiment pipeline.
* **Never fatal.** Every failure path — provider down, timeout, malformed JSON,
  label outside the taxonomy — degrades to a well-formed record with
  ``risk_score`` 0 and ``recommended_action`` "Human Review". The intelligence
  stage can never break a sentiment response.
* **Deterministic where it can be.** Posts with no analyzable content at all
  (empty after cleaning, emoji-only, media placeholder) are resolved by
  :func:`triage` without a model call. Everything else goes to the model with
  precomputed edge-case *signals* attached, so the model weighs them rather than
  having to rediscover them.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import config
from src.script_detection import LATIN_RANGE, UNICODE_RANGES

logger = logging.getLogger("benchmark.intelligence")

_ALLOWED_INTENTS = frozenset(config.INTENT_LABELS) | {config.UNKNOWN_LABEL}
_ALLOWED_CATEGORIES = frozenset(config.CATEGORY_LABELS) | {config.UNKNOWN_LABEL}
_ALLOWED_ACTIONS = frozenset(config.ACTION_LABELS)
_ALLOWED_EVIDENCE = frozenset(("high", "medium", "low"))


# --------------------------------------------------------------------------- #
# Guard rails — sent as the system prompt on every single invocation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""\
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

YOUR TASKS — intent, category, contextual risk, reasoning, summary, action.

intent — exactly one of:
  {", ".join(config.INTENT_LABELS)}, or "{config.UNKNOWN_LABEL}"

category — exactly one of:
  {", ".join(config.CATEGORY_LABELS)}, or "{config.UNKNOWN_LABEL}"

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
- Very short, ambiguous or incomplete posts: prefer "{config.UNKNOWN_LABEL}",
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
  category / intent  -> "{config.UNKNOWN_LABEL}"
  risk_score         -> 0
  reasoning          -> explain exactly what is missing
  recommended_action -> "Human Review"
  evidence_confidence-> "low"

Return only this JSON object:
{{"category": "...", "intent": "...", "risk_score": 0, "reasoning": "...", \
"summary": "...", "recommended_action": "...", "evidence_confidence": "..."}}
"""

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": sorted(_ALLOWED_CATEGORIES)},
        "intent": {"type": "string", "enum": sorted(_ALLOWED_INTENTS)},
        "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
        "recommended_action": {"type": "string", "enum": sorted(_ALLOWED_ACTIONS)},
        "evidence_confidence": {"type": "string", "enum": sorted(_ALLOWED_EVIDENCE)},
    },
    "required": [
        "category", "intent", "risk_score", "reasoning", "summary",
        "recommended_action", "evidence_confidence",
    ],
}


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

    def to_dict(self) -> dict:
        return asdict(self)


def _insufficient(
    reasoning: str, signals: list[str], source: str, action: str = "Human Review"
) -> IntelligenceResult:
    """The mandated shape for 'not enough evidence' — used by both triage and
    every failure path, so an unreadable post and an unreachable Ollama produce
    records a consumer can treat identically."""
    return IntelligenceResult(
        category=config.UNKNOWN_LABEL,
        intent=config.UNKNOWN_LABEL,
        risk_score=0,
        reasoning=reasoning,
        summary="No intelligence assessment could be produced for this post.",
        recommended_action=action,
        evidence_confidence="low",
        signals=signals,
        source=source,
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
    """Fraction of alphabetic characters in the minority script (Latin vs Indic).

    0.0 means single-script. Uses the same Unicode ranges as the pipeline's
    script detection so "code-mixed" means the same thing in both places.
    """
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
    """Cheap, deterministic edge-case flags attached to the prompt.

    These do not decide anything — the model is told to weigh them. Computing
    them here rather than asking the model to notice them makes the edge-case
    handling auditable and consistent across posts.
    """
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
    # Not English, but translation never ran -> no FLORES mapping for the
    # detected language, so english_text is really the untranslated source.
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
    # Long runs of isolated single characters are the usual OCR signature.
    if len(words) >= 6 and sum(1 for w in words if len(w) == 1) / len(words) > 0.4:
        signals.append("possible_ocr_noise")
    return signals


def triage(result: dict, signals: list[str]) -> IntelligenceResult | None:
    """Resolve posts with no analyzable content without calling the model.

    Returns ``None`` when the post should go to the provider. Only genuinely
    contentless posts are short-circuited here — everything ambiguous is the
    model's call, with the signals attached.
    """
    original = str(result.get("post_text") or "")
    english = str(result.get("english_text") or "")

    if not original.strip():
        return _insufficient(
            "The post is empty or contains only whitespace; there is no content to assess.",
            signals, "triage", action="Ignore",
        )
    if _MEDIA_RE.match(original.strip()):
        return _insufficient(
            "The post is a media placeholder with no accompanying text. The "
            "attached media itself was not available for assessment.",
            signals, "triage",
        )
    stripped = _URL_RE.sub(" ", original)
    if not any(ch.isalnum() for ch in stripped):
        # No letters or digits once links are removed: emoji-only, punctuation-
        # only, or link-only. Sentiment may still exist; intelligence cannot.
        reason = (
            "The post contains no textual content once links are removed "
            "(emoji, punctuation or URL only), so intent, category and "
            "contextual risk cannot be assessed from text."
        )
        return _insufficient(reason, signals, "triage", action="Ignore")
    if not english.strip():
        return _insufficient(
            "The pipeline produced no English text for this post, so no "
            "assessment can be made from the translation.",
            signals, "triage",
        )
    return None


def build_payload(result: dict, signals: list[str]) -> dict:
    """The structured user message — pipeline facts, never bare text.

    Long posts are truncated head+tail so the opening framing and any closing
    call to action both survive; the `truncated` signal tells the model.
    """
    def clip(text: str) -> str:
        limit = config.INTELLIGENCE_MAX_TEXT_CHARS
        if len(text) <= limit:
            return text
        head = text[: int(limit * 0.7)]
        tail = text[-int(limit * 0.3) :]
        return f"{head}\n...[truncated]...\n{tail}"

    return {
        "original_text": clip(str(result.get("post_text") or "")),
        "language": result.get("language"),
        "english_text": clip(str(result.get("english_text") or "")),
        "sentiment": result.get("sentiment"),
        "confidence": result.get("confidence"),
        "was_translated": bool(result.get("was_translated")),
        "was_transliterated": bool(result.get("was_transliterated")),
        "signals": signals,
    }


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class IntelligenceProvider(ABC):
    """Extension point. Register new backends in :data:`PROVIDERS`.

    Implementations return the raw decoded JSON object; validation, taxonomy
    clamping and fallbacks are handled centrally by :class:`IntelligenceAnalyzer`,
    so a new provider only has to make the call.
    """

    name: str = "provider"

    @abstractmethod
    def generate(self, payload: dict) -> dict:
        """Return the model's parsed JSON object, or raise."""

    @abstractmethod
    def describe(self) -> dict:
        """Static config, surfaced by /health."""

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
        import requests  # transitively present via huggingface_hub; pinned in requirements

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self._use_schema = use_schema
        self._session = requests.Session()
        self._lock = threading.Lock()  # guards the schema-support downgrade

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "json_schema": self._use_schema,
        }

    def health(self) -> bool:
        """True when the Ollama host answers. Never raises."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as exc:
            logger.warning("Ollama health check failed for %s: %s", self.base_url, exc)
            return False

    def _body(self, payload: dict, use_schema: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "seed": config.SEED,
            },
        }
        # A schema constrains decoding; "json" only guarantees well-formedness.
        body["format"] = RESPONSE_SCHEMA if use_schema else "json"
        return body

    def generate(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            use_schema = self._use_schema
            try:
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json=self._body(payload, use_schema),
                    timeout=self.timeout_s,
                )
                if use_schema and response.status_code == 400:
                    # Older Ollama rejects a schema in `format`. Downgrade once,
                    # for the life of the process, and retry immediately.
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
                        json=self._body(payload, False),
                        timeout=self.timeout_s,
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
        raise RuntimeError(f"Ollama request failed after {self.retries + 1} attempt(s): {last_error}")

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass


PROVIDERS: dict[str, type[IntelligenceProvider]] = {
    OllamaProvider.name: OllamaProvider,
}


def _parse_json_object(content: str) -> dict:
    """Decode the model's reply, tolerating a stray code fence or prose.

    Rule 10 forbids both, but a fallback beats discarding an otherwise good
    assessment because the model wrapped it in ```json.
    """
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

    def analyze_one(self, result: dict) -> IntelligenceResult:
        """Never raises — every failure becomes an 'insufficient evidence' record."""
        signals = derive_signals(result)
        short_circuit = triage(result, signals)
        if short_circuit is not None:
            return short_circuit

        started = time.perf_counter()
        try:
            raw = self.provider.generate(build_payload(result, signals))
        except Exception as exc:
            logger.error("Intelligence provider failed: %s", exc)
            record = _insufficient(
                f"The intelligence provider could not be reached or returned an "
                f"unusable response ({type(exc).__name__}). The deterministic "
                f"sentiment result is unaffected and remains valid.",
                signals, "error",
            )
            record.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
            record.model = getattr(self.provider, "model", "")
            return record

        record = self._validate(raw, signals)
        record.latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        record.model = getattr(self.provider, "model", "")
        return record

    def analyze_batch(self, results: list[dict]) -> list[IntelligenceResult]:
        """Order-preserving. Duplicate posts share one provider call."""
        if not results:
            return []

        # Dedupe on the exact text the model would see.
        first_index: dict[tuple, int] = {}
        todo: list[int] = []
        reuse: dict[int, int] = {}
        for i, result in enumerate(results):
            key = (result.get("post_text"), result.get("english_text"), result.get("sentiment"))
            if key in first_index:
                reuse[i] = first_index[key]
                continue
            first_index[key] = i
            todo.append(i)

        workers = max(1, min(config.OLLAMA_CONCURRENCY, len(todo)))
        started = time.perf_counter()
        computed: dict[int, IntelligenceResult] = {}
        if workers == 1:
            for i in todo:
                computed[i] = self.analyze_one(results[i])
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, record in zip(todo, pool.map(lambda j: self.analyze_one(results[j]), todo)):
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
            "%d resolved without a model call, %d failed",
            len(results), (time.perf_counter() - started) * 1000.0,
            len(todo), len(reuse),
            sum(1 for r in out if r.source == "triage"), errors,
        )
        return out

    def _validate(self, raw: dict, signals: list[str]) -> IntelligenceResult:
        """Clamp the model's reply onto the contract.

        An out-of-taxonomy label becomes "Unknown" rather than being passed
        through, so a downstream consumer can rely on the enum.
        """
        def pick(key: str, allowed: frozenset[str], default: str) -> str:
            value = str(raw.get(key, "") or "").strip()
            if value in allowed:
                return value
            # Tolerate case drift ("monitor" -> "Monitor") before giving up.
            for candidate in allowed:
                if candidate.lower() == value.lower():
                    return candidate
            if value:
                logger.warning("Model returned %s=%r, outside the taxonomy.", key, value)
            return default

        category = pick("category", _ALLOWED_CATEGORIES, config.UNKNOWN_LABEL)
        intent = pick("intent", _ALLOWED_INTENTS, config.UNKNOWN_LABEL)
        action = pick("recommended_action", _ALLOWED_ACTIONS, "Human Review")
        evidence = pick("evidence_confidence", _ALLOWED_EVIDENCE, "low")

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

        return IntelligenceResult(
            category=category,
            intent=intent,
            risk_score=risk,
            reasoning=reasoning,
            summary=summary,
            recommended_action=action,
            evidence_confidence=evidence,
            signals=signals,
            source="provider",
        )

    def close(self) -> None:
        self.provider.close()
