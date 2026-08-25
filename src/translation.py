"""Stage 1 — Indian-language -> English translation pipelines.

Supported families:
    * ``indictrans2`` — AI4Bharat IndicTrans2. Uses ``IndicTransToolkit`` for
      the recommended pre/post-processing when installed; otherwise falls back
      to plain "<src> <tgt> sentence" tag prefixing (slightly lower quality,
      logged as a warning).
    * ``nllb``        — Meta NLLB-200 (``src_lang`` + forced BOS token).
    * ``madlad``      — Google MADLAD-400 ("<2en> " prefix).

Genuine English posts (English content, typed in English) bypass this stage
entirely — no translation needed. Everything else routes through the model,
including romanized Indic languages ("Hinglish" etc.), which by the time
they reach this module have already been converted to native script by
src/transliteration.py (IndicXlit) upstream in pipeline.py. Only a post whose
detected language has no FLORES mapping at all (detection failed / returned
'unknown') passes through unchanged, since there is no source-language token
to translate from.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    FLORES_CODES,
    GENERATION_MAX_TIME_S,
    INFERENCE_HARD_TIMEOUT_S,
    TARGET_FLORES,
    TRANSLATION_CACHE_SIZE,
    TRANSLATION_LENGTH_MARGIN,
    TRANSLATION_LENGTH_RATIO,
    TRANSLATION_MAX_LENGTH,
    TRANSLATION_NUM_BEAMS,
    TRANSLATION_OUTPUT_MAX_RATIO,
    TRANSLATION_REPEAT_TOKEN_RATIO,
    TranslationConfig,
)
from src.inference_watchdog import guarded_model_call
from src.transliteration import source_polarity_tokens
from src.utils import chunked, get_gpu_peak_mb, get_model_size_mb, reset_gpu_peak

logger = logging.getLogger("benchmark.translation")


class TranslatorLoadError(RuntimeError):
    """A translation model could not be loaded; the message says how to fix it."""


class TranslationError(RuntimeError):
    """A translation request could not produce a safe English result."""


class TranslationPreprocessError(TranslationError):
    """Input could not be prepared safely for the selected translation model."""


class TranslationOutputError(TranslationError):
    """Neither translator produced a credible English result."""


def _load_error_hint(cfg: TranslationConfig, exc: Exception) -> str:
    """Turn a raw HF/transformers load failure into an actionable message."""
    text = f"{type(exc).__name__}: {exc}"
    url = f"https://huggingface.co/{cfg.hf_id}"
    if "401" in text or "Unauthorized" in text or "gated" in text.lower():
        return (
            f"{cfg.display_name} ({cfg.hf_id}) is a GATED model and you are not "
            f"authenticated (401).\nFix: run `huggingface-cli login` with a token "
            f"from https://huggingface.co/settings/tokens, and make sure you have "
            f"requested access at {url}\nOriginal error: {text}"
        )
    if "403" in text or "Forbidden" in text:
        return (
            f"Your HuggingFace token is valid but access to {cfg.hf_id} has not "
            f"been granted yet (403).\nFix: open {url}, click 'Agree and access "
            f"repository' (approval can take a moment), then re-run.\n"
            f"Original error: {text}"
        )
    if "transformers.onnx" in text or "past_key_values" in text or "Cache" in text:
        return (
            f"{cfg.display_name} failed to load due to a transformers version "
            f"incompatibility — its custom code needs the legacy API.\n"
            f"Fix: pip install 'transformers==4.40.2' (v5 removed "
            f"transformers.onnx; >=4.5x changed the KV-cache API).\n"
            f"Original error: {text}"
        )
    return f"Failed to load {cfg.display_name} ({cfg.hf_id}).\nOriginal error: {text}"


@dataclass
class TranslationResult:
    """Per-post translations plus aggregate runtime statistics."""

    texts: list[str]         # English output, original order
    times_ms: np.ndarray     # amortized per-post translation time (0 for pass-through)
    total_time_s: float      # wall-clock time for the whole dataset
    gpu_peak_mb: float
    translated_mask: np.ndarray  # True where the post was actually translated
    n_cached: int = 0        # posts served from the translation cache (times_ms 0)
    backends: list[str] | None = None  # bypass | passthrough | indictrans2 | nllb | ...
    truncated_mask: np.ndarray | None = None


def _arabic_script_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    arabic = sum("\u0600" <= char <= "\u06ff" for char in letters)
    return arabic / len(letters)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")


def split_translation_chunks(text: str) -> list[str]:
    """Split only when sentence boundaries exist.

    IndicTrans2 is trained at sentence level (Gala et al., arXiv:2305.16307).
    This repo still translates the whole post by default; chunks are used
    only when the tokenizer would truncate a long multi-sentence post.
    """
    text = str(text or "").strip()
    if not text:
        return [text]
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(parts) >= 2:
        return parts
    return [text]


# Closed English polarity lexicons for the usable-gate check.
# Inspired by sentiment-aware MT evaluation (SAM / SentiWord-style prior
# polarities; Orăsan & Tantawy, arXiv:2109.14895): judge polarity class
# preservation, not exact token identity. Paraphrases ("worst"→"bad",
# "good"→"well") must pass; polarity loss or reversal must fail.
# No extra neural QE / LLM — research shows those help overall quality but
# are unnecessary for this narrow production gate.
_POSITIVE_ENGLISH = frozenset(
    {
        "good", "well", "best", "better", "super", "great", "excellent",
        "amazing", "awesome", "nice", "love", "useful", "fine", "perfect",
        "wonderful", "happy", "glad", "positive",
    }
)
_NEGATIVE_ENGLISH = frozenset(
    {
        "bad", "worst", "worse", "terrible", "horrible", "poor", "hate",
        "useless", "awful", "negative", "disappointing", "failed", "failure",
    }
)
_NEGATION_ENGLISH = frozenset(
    {
        "not", "no", "never", "neither", "nor", "without", "none", "cannot",
        "cant",
    }
)
_SOURCE_TOKEN_POLARITY = {
    "good": "pos",
    "best": "pos",
    "super": "pos",
    "great": "pos",
    "excellent": "pos",
    "love": "pos",
    "useful": "pos",
    "better": "pos",
    "bad": "neg",
    "worst": "neg",
    "worse": "neg",
    "terrible": "neg",
    "horrible": "neg",
    "poor": "neg",
    "hate": "neg",
    "useless": "neg",
    "not": "negation",
    "no": "negation",
    "never": "negation",
}
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _english_word_cores(text: str) -> list[str]:
    cores: list[str] = []
    for match in _ENGLISH_WORD_RE.finditer(str(text or "")):
        raw = match.group(0).casefold()
        if raw.endswith("n't"):
            cores.append("not")
            stem = raw[:-3]
            if stem and stem not in {"wo", "ca", "sha", "ai"}:
                cores.append(stem)
            elif stem == "ca":
                cores.append("can")
            continue
        cores.append(raw)
    return cores


def _effective_polarity(has_pos: bool, has_neg: bool, has_negation: bool) -> str:
    """Collapse lexicon hits into a coarse polarity label.

    Negation flips a lone positive or lone negative cue ("not good" → neg,
    "not bad" → pos). Mixed cues stay mixed; no cues → none.
    """
    if has_negation:
        if has_pos and not has_neg:
            return "neg"
        if has_neg and not has_pos:
            return "pos"
        if has_pos and has_neg:
            return "mixed"
        return "negation_only"
    if has_pos and has_neg:
        return "mixed"
    if has_pos:
        return "pos"
    if has_neg:
        return "neg"
    return "none"


def polarity_is_preserved(source: str, translated: str) -> bool:
    """True when English polarity cues survive paraphrase (not exact match).

    Only applies when the *source* carries English polarity/negation tokens
    (code-mix). Pure Indic polarity is left to the MT models — this gate does
    not invent a second sentiment classifier.
    """
    src_tokens = source_polarity_tokens(source)
    if not src_tokens:
        return True

    src_pos = any(_SOURCE_TOKEN_POLARITY.get(token) == "pos" for token in src_tokens)
    src_neg = any(_SOURCE_TOKEN_POLARITY.get(token) == "neg" for token in src_tokens)
    src_negation = any(
        _SOURCE_TOKEN_POLARITY.get(token) == "negation" for token in src_tokens
    )
    # Bare English negation with no sentiment adjective still requires a
    # negation cue in the translation (negation drop flips meaning).
    if src_negation and not src_pos and not src_neg:
        tgt_cores = _english_word_cores(translated)
        return any(
            core in _NEGATION_ENGLISH or core.endswith("n't")
            for core in tgt_cores
        )

    src_eff = _effective_polarity(src_pos, src_neg, src_negation)
    if src_eff not in {"pos", "neg"}:
        return True

    tgt_cores = _english_word_cores(translated)
    tgt_pos = any(core in _POSITIVE_ENGLISH for core in tgt_cores)
    tgt_neg = any(core in _NEGATIVE_ENGLISH for core in tgt_cores)
    tgt_negation = any(
        core in _NEGATION_ENGLISH or core.endswith("n't") for core in tgt_cores
    )
    tgt_eff = _effective_polarity(tgt_pos, tgt_neg, tgt_negation)

    if tgt_eff == "none":
        return False  # polarity loss
    if tgt_eff in {"pos", "neg"} and tgt_eff != src_eff:
        return False  # polarity reversal
    return True


def translation_is_usable(source: str, translated: str) -> bool:
    """Reject outputs that cannot be a credible English translation."""
    source = str(source or "").strip()
    translated = str(translated or "").strip()
    if not translated:
        return False
    if source.casefold() == translated.casefold():
        return False
    letters = [char for char in translated if char.isalpha()]
    if not letters:
        return False
    latin_letters = sum(
        ("a" <= char.casefold() <= "z") for char in letters
    )
    if latin_letters / len(letters) < 0.6:
        return False
    if _arabic_script_ratio(source) >= 0.5 and _arabic_script_ratio(translated) >= 0.2:
        return False
    if len(translated) > max(256, int(len(source) * TRANSLATION_OUTPUT_MAX_RATIO)):
        return False

    tokens = translated.casefold().split()
    if len(tokens) >= 8:
        most_common = Counter(tokens).most_common(1)[0][1]
        if most_common / len(tokens) >= TRANSLATION_REPEAT_TOKEN_RATIO:
            return False
    if not polarity_is_preserved(source, translated):
        return False
    return True


class Translator:
    """One translation pipeline wrapped for benchmarking.

    Thread safety: :meth:`translate` holds an instance lock for its whole body.
    The generation loop mutates model/tokenizer state (NLLB selects its source
    language through ``tokenizer.src_lang``), so two threads translating
    different languages at once would otherwise read each other's source-language
    token and silently emit translations from the wrong language. The lock plus
    :meth:`_source_language` — which always restores the previous value — keeps
    that state from escaping a single call.
    """

    def __init__(self, cfg: TranslationConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        trust = cfg.family == "indictrans2"
        logger.info("Loading translator %s (%s)", cfg.display_name, cfg.hf_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=trust)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.hf_id, trust_remote_code=trust)
        except Exception as exc:
            raise TranslatorLoadError(_load_error_hint(cfg, exc)) from exc
        self.model.to(device)
        self.model.eval()

        self._lock = threading.Lock()
        # (lang, source text) -> translation. Bounded LRU; guarded by _lock,
        # which is held for the whole of translate().
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()

        self._indic_processor = None
        self._indic_processor_broken_langs: set[str] = set()
        self._urdu_preprocessing_ready = cfg.family != "indictrans2"
        if cfg.family == "indictrans2":
            try:
                from IndicTransToolkit.processor import IndicProcessor

                self._indic_processor = IndicProcessor(inference=True)
            except Exception as exc:
                raise TranslatorLoadError(
                    "IndicTransToolkit is required for safe IndicTrans2 preprocessing"
                ) from exc
            self._validate_urdu_preprocessor()

    @property
    def size_mb(self) -> float:
        return get_model_size_mb(self.model)

    @property
    def preprocessing_mode(self) -> str:
        """Which preprocessing path is live — reported by the service's /health
        so a silent degradation to tag prefixing is visible without log access."""
        if self.cfg.family != "indictrans2":
            return "native"
        return "IndicTransToolkit" if self._indic_processor is not None else "plain-tag-prefix"

    @property
    def urdu_preprocessing_ready(self) -> bool:
        return self._urdu_preprocessing_ready

    def _validate_urdu_preprocessor(self) -> None:
        """Fail startup if the active ``indicnlp`` package cannot process Urdu.

        Installing both ``indic-nlp-library`` and ``indic-nlp-library-itt`` is
        unsafe because they own identical import paths. This probe catches that
        environment error before the service reports healthy.
        """
        try:
            prepared = self._indic_processor.preprocess_batch(
                ["یہ ایک آزمائشی جملہ ہے۔"],
                src_lang="urd_Arab",
                tgt_lang=TARGET_FLORES,
            )
        except Exception as exc:
            raise TranslatorLoadError(
                "Urdu preprocessing is unavailable. Install only "
                "indic-nlp-library-itt==0.1.1; remove indic-nlp-library and "
                "standalone urduhack."
            ) from exc
        if not prepared or not str(prepared[0]).strip():
            raise TranslatorLoadError("Urdu preprocessing returned an empty result")
        self._urdu_preprocessing_ready = True

    # ------------------------------------------------------------------ #
    # Translation cache
    # ------------------------------------------------------------------ #
    def _cache_get(self, lang: str, text: str) -> str | None:
        if TRANSLATION_CACHE_SIZE <= 0:
            return None
        hit = self._cache.get((lang, text))
        if hit is not None:
            self._cache.move_to_end((lang, text))
        return hit

    def _cache_put(self, lang: str, text: str, translation: str) -> None:
        if TRANSLATION_CACHE_SIZE <= 0:
            return
        self._cache[(lang, text)] = translation
        self._cache.move_to_end((lang, text))
        while len(self._cache) > TRANSLATION_CACHE_SIZE:
            self._cache.popitem(last=False)

    # ------------------------------------------------------------------ #
    def _prepare_batch(self, batch: list[str], src_flores: str) -> list[str] | dict:
        if self.cfg.family == "indictrans2":
            use_processor = (
                self._indic_processor is not None
                and src_flores not in self._indic_processor_broken_langs
            )
            if use_processor:
                try:
                    return self._indic_processor.preprocess_batch(
                        batch, src_lang=src_flores, tgt_lang=TARGET_FLORES
                    )
                except Exception as exc:
                    if src_flores == "urd_Arab":
                        raise TranslationPreprocessError(
                            "IndicTransToolkit failed to preprocess Urdu"
                        ) from exc
                    # Preserve the established lower-quality fallback for
                    # languages whose optional normalizer fails. Urdu is
                    # excluded because plain tags can cause degenerate decode.
                    self._indic_processor_broken_langs.add(src_flores)
                    logger.warning(
                        "IndicTransToolkit preprocessing failed for %s (%s) — "
                        "falling back to plain tag prefixing for this language.",
                        src_flores, exc,
                    )
            return [f"{src_flores} {TARGET_FLORES} {text}" for text in batch]
        if self.cfg.family == "madlad":
            return [f"<2en> {text}" for text in batch]
        return batch  # nllb: language handled via tokenizer.src_lang

    @contextmanager
    def _source_language(self, src_flores: str):
        """Scope NLLB's ``tokenizer.src_lang`` to one batch and always restore it.

        Only NLLB selects its source language this way; every other family
        encodes it in the input text, so this is a no-op for them. Restoring
        matters because the attribute is shared process-wide state on a
        long-lived tokenizer — leaving the last batch's language set is how a
        later call ends up translating from the wrong source language.
        """
        if self.cfg.family != "nllb":
            yield
            return
        previous = getattr(self.tokenizer, "src_lang", None)
        self.tokenizer.src_lang = src_flores
        try:
            yield
        finally:
            self.tokenizer.src_lang = previous

    def _generate_kwargs(self, source_tokens: int) -> dict:
        """Generation budget for one batch.

        ``max_new_tokens`` is sized from the longest source in the batch rather
        than pinned at TRANSLATION_MAX_LENGTH: a translation is never many times
        longer than its source, so the flat budget only ever paid off for a
        decode that had already gone wrong (looping until it hit the cap). See
        TRANSLATION_LENGTH_RATIO in config.py.
        """
        budget = TRANSLATION_MAX_LENGTH
        if TRANSLATION_LENGTH_RATIO > 0:
            budget = min(
                TRANSLATION_MAX_LENGTH,
                int(source_tokens * TRANSLATION_LENGTH_RATIO) + TRANSLATION_LENGTH_MARGIN,
            )
        kwargs = {
            "max_new_tokens": max(1, budget),
            "num_beams": TRANSLATION_NUM_BEAMS,
            "do_sample": False,
            "max_time": GENERATION_MAX_TIME_S,
        }
        if self.cfg.family == "nllb":
            kwargs["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(TARGET_FLORES)
        return kwargs

    def _postprocess(self, decoded: list[str]) -> list[str]:
        if self.cfg.family == "indictrans2" and self._indic_processor is not None:
            return self._indic_processor.postprocess_batch(decoded, lang=TARGET_FLORES)
        return [text.strip() for text in decoded]

    # ------------------------------------------------------------------ #
    def translate(
        self, texts: list[str], languages: list[str], batch_size: int = 8
    ) -> TranslationResult:
        """Translate every non-English post; keep the original order.

        Serialized on the instance lock — see the class docstring.
        """
        with self._lock:
            return self._translate_locked(texts, languages, batch_size)

    def _translate_locked(
        self, texts: list[str], languages: list[str], batch_size: int
    ) -> TranslationResult:
        n = len(texts)
        out_texts: list[str] = list(texts)  # pass-through default
        times_ms = np.zeros(n)
        translated = np.zeros(n, dtype=bool)
        backends = ["passthrough"] * n
        truncated = np.zeros(n, dtype=bool)

        # Group indices by source language so each batch is monolingual.
        # 'en' is bypassed: genuine English (content AND script both English)
        # needs no translation. Every other language routes through the model,
        # INCLUDING romanized Indic languages once script_detection +
        # transliteration.py have already converted them to native script
        # upstream in pipeline.py — 'hi' text arriving here is native Devanagari
        # whether the original post was typed in Devanagari or Latin letters.
        # Only a language with no FLORES mapping at all (detection failed /
        # returned 'unknown') has no source token to translate from.
        #
        # Posts already in the cache are resolved here and never reach the
        # model, so a retried request pays nothing for work the cancelled one
        # already completed.
        groups: dict[str, list[int]] = defaultdict(list)
        skipped_langs: set[str] = set()
        n_cached = 0
        for idx, lang in enumerate(languages):
            if lang == "en":
                backends[idx] = "bypass"
                continue
            if lang not in FLORES_CODES:
                skipped_langs.add(lang)
                continue
            cached = self._cache_get(lang, texts[idx])
            if cached is not None:
                out_texts[idx] = cached
                translated[idx] = True
                n_cached += 1
                backends[idx] = self.cfg.key
                continue
            groups[lang].append(idx)
        if skipped_langs:
            logger.warning(
                "%s: no FLORES mapping for detected language(s) %s — those posts "
                "are passed through untranslated.",
                self.cfg.display_name, sorted(skipped_langs),
            )

        reset_gpu_peak(self.device)
        total_start = time.perf_counter()
        total_posts = sum(len(v) for v in groups.values())
        progress = tqdm(total=total_posts, desc=f"Translating [{self.cfg.display_name}]", unit="post")

        with torch.inference_mode():
            for lang, indices in groups.items():
                src_flores = FLORES_CODES[lang]
                with self._source_language(src_flores):
                    for index_batch in chunked(indices, batch_size):
                        batch = [texts[i] for i in index_batch]
                        prepared = self._prepare_batch(batch, src_flores)
                        encoded = self.tokenizer(
                            prepared,
                            truncation=True,
                            padding=True,
                            max_length=TRANSLATION_MAX_LENGTH,
                            return_tensors="pt",
                        )
                        token_counts = encoded["attention_mask"].sum(dim=1)
                        for local_i, original_i in enumerate(index_batch):
                            if int(token_counts[local_i]) >= TRANSLATION_MAX_LENGTH:
                                truncated[original_i] = True

                        source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                        generate_kwargs = self._generate_kwargs(source_tokens)

                        def generate_batch():
                            encoded_on_device = encoded.to(self.device)
                            if self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            t0 = time.perf_counter()
                            generated_batch = self.model.generate(
                                **encoded_on_device, **generate_kwargs
                            )
                            if self.device.type == "cuda":
                                torch.cuda.synchronize(self.device)
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            return generated_batch, elapsed

                        generated, elapsed_ms = guarded_model_call(
                            generate_batch, INFERENCE_HARD_TIMEOUT_S
                        )

                        # A decode that used its entire budget almost certainly
                        # never emitted EOS — the signature of a runaway decode.
                        # Log it: this is the evidence for whether slow posts are
                        # slow because of the model or because of degenerate output.
                        new_tokens = int(generated.shape[-1])
                        if new_tokens >= generate_kwargs["max_new_tokens"]:
                            logger.warning(
                                "%s [%s]: decode hit the %d-token budget for a "
                                "%d-token source (%d post(s), %.0f ms) — output is "
                                "likely truncated or degenerate.",
                                self.cfg.display_name, lang,
                                generate_kwargs["max_new_tokens"], source_tokens,
                                len(index_batch), elapsed_ms,
                            )
                        else:
                            logger.debug(
                                "%s [%s]: %d post(s), %d source tokens -> %d generated, "
                                "%.0f ms (%.0f ms/post)",
                                self.cfg.display_name, lang, len(index_batch),
                                source_tokens, new_tokens, elapsed_ms,
                                elapsed_ms / len(index_batch),
                            )

                        decoded = self.tokenizer.batch_decode(
                            generated, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        for i, translation in zip(index_batch, self._postprocess(decoded)):
                            out_texts[i] = translation if translation else texts[i]
                            times_ms[i] = elapsed_ms / len(index_batch)
                            translated[i] = True
                            backends[i] = self.cfg.key
                            if translation:
                                self._cache_put(lang, texts[i], translation)
                        for original_i in index_batch:
                            if not truncated[original_i]:
                                continue
                            chunks = split_translation_chunks(texts[original_i])
                            if len(chunks) < 2:
                                continue
                            piece_out: list[str] = []
                            chunk_truncated = False
                            for chunk in chunks:
                                prepared_chunk = self._prepare_batch([chunk], src_flores)
                                encoded_chunk = self.tokenizer(
                                    prepared_chunk,
                                    truncation=True,
                                    padding=True,
                                    max_length=TRANSLATION_MAX_LENGTH,
                                    return_tensors="pt",
                                )
                                if int(encoded_chunk["attention_mask"].sum()) >= TRANSLATION_MAX_LENGTH:
                                    chunk_truncated = True
                                    break
                                source_tokens = int(encoded_chunk["attention_mask"].sum())
                                generate_kwargs = self._generate_kwargs(source_tokens)

                                encoded_chunk_local = encoded_chunk
                                generate_kwargs_local = generate_kwargs

                                def generate_chunk(
                                    encoded_chunk=encoded_chunk_local,
                                    generate_kwargs=generate_kwargs_local,
                                ):
                                    encoded_on_device = encoded_chunk.to(self.device)
                                    generated_chunk = self.model.generate(
                                        **encoded_on_device, **generate_kwargs
                                    )
                                    return generated_chunk

                                generated_chunk = guarded_model_call(
                                    generate_chunk, INFERENCE_HARD_TIMEOUT_S
                                )
                                decoded_chunk = self.tokenizer.batch_decode(
                                    generated_chunk, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True,
                                )
                                piece_out.extend(self._postprocess(decoded_chunk))
                            if piece_out and not chunk_truncated:
                                joined = " ".join(part for part in piece_out if part)
                                if joined:
                                    out_texts[original_i] = joined
                                    truncated[original_i] = False
                                    self._cache_put(lang, texts[original_i], joined)
                        progress.update(len(index_batch))
        progress.close()

        total_time = time.perf_counter() - total_start
        logger.info(
            "%s translated %d/%d posts in %.1fs (%d from cache)",
            self.cfg.display_name, int(translated.sum()), n, total_time, n_cached,
        )
        return TranslationResult(
            texts=out_texts,
            times_ms=times_ms,
            total_time_s=total_time,
            gpu_peak_mb=get_gpu_peak_mb(self.device),
            translated_mask=translated,
            n_cached=n_cached,
            backends=backends,
            truncated_mask=truncated,
        )

    def free(self) -> None:
        """Release the model. Idempotent — safe to call more than once, which
        matters now that the service frees the pipeline on shutdown as well."""
        self.model = None
        self.tokenizer = None
        self._cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
