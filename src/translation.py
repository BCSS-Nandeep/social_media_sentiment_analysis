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
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    FLORES_CODES,
    TARGET_FLORES,
    TRANSLATION_CACHE_SIZE,
    TRANSLATION_LENGTH_MARGIN,
    TRANSLATION_LENGTH_RATIO,
    TRANSLATION_MAX_LENGTH,
    TRANSLATION_NUM_BEAMS,
    TranslationConfig,
)
from src.utils import chunked, get_gpu_peak_mb, get_model_size_mb, reset_gpu_peak

logger = logging.getLogger("benchmark.translation")


class TranslatorLoadError(RuntimeError):
    """A translation model could not be loaded; the message says how to fix it."""


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
        if cfg.family == "indictrans2":
            try:
                from IndicTransToolkit.processor import IndicProcessor

                self._indic_processor = IndicProcessor(inference=True)
            except Exception as exc:
                logger.warning(
                    "IndicTransToolkit unavailable (%s) — falling back to plain "
                    "tag prefixing for IndicTrans2 (slightly lower quality).",
                    exc,
                )

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
                    # e.g. IndicTransToolkit's Urdu normalizer needs the
                    # `urduhack` package, which isn't installed here (it
                    # transitively pulls in TensorFlow, which segfaults
                    # alongside the fairseq-based transliteration stage
                    # already running in this process). Degrade to plain tag
                    # prefixing for this language only — every other language
                    # keeps the better IndicTransToolkit preprocessing.
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
                continue
            if lang not in FLORES_CODES:
                skipped_langs.add(lang)
                continue
            cached = self._cache_get(lang, texts[idx])
            if cached is not None:
                out_texts[idx] = cached
                translated[idx] = True
                n_cached += 1
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
                        ).to(self.device)

                        source_tokens = int(encoded["attention_mask"].sum(dim=1).max())
                        generate_kwargs = self._generate_kwargs(source_tokens)

                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        t0 = time.perf_counter()
                        generated = self.model.generate(**encoded, **generate_kwargs)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0

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
                            if translation:
                                self._cache_put(lang, texts[i], translation)
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
        )

    def free(self) -> None:
        """Release the model. Idempotent — safe to call more than once, which
        matters now that the service frees the pipeline on shutdown as well."""
        self.model = None
        self.tokenizer = None
        self._cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
