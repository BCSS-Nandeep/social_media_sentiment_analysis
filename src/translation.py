"""Stage 1 — Indian-language -> English translation pipelines.

Supported families:
    * ``indictrans2`` — AI4Bharat IndicTrans2. Uses ``IndicTransToolkit`` for
      the recommended pre/post-processing when installed; otherwise falls back
      to plain "<src> <tgt> sentence" tag prefixing (slightly lower quality,
      logged as a warning).
    * ``nllb``        — Meta NLLB-200 (``src_lang`` + forced BOS token).
    * ``madlad``      — Google MADLAD-400 ("<2en> " prefix).

Posts already in English — and posts whose detected language has no FLORES
mapping (e.g. misdetected romanized code-mix) — are passed through unchanged
with zero translation latency.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    FLORES_CODES,
    TARGET_FLORES,
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


class Translator:
    """One translation pipeline wrapped for benchmarking."""

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

        self._indic_processor = None
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

    # ------------------------------------------------------------------ #
    def _prepare_batch(self, batch: list[str], src_flores: str) -> list[str] | dict:
        if self.cfg.family == "indictrans2":
            if self._indic_processor is not None:
                return self._indic_processor.preprocess_batch(
                    batch, src_lang=src_flores, tgt_lang=TARGET_FLORES
                )
            return [f"{src_flores} {TARGET_FLORES} {text}" for text in batch]
        if self.cfg.family == "madlad":
            return [f"<2en> {text}" for text in batch]
        return batch  # nllb: language handled via tokenizer.src_lang

    def _generate_kwargs(self) -> dict:
        kwargs = {
            "max_length": TRANSLATION_MAX_LENGTH,
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
        """Translate every non-English post; keep the original order."""
        n = len(texts)
        out_texts: list[str] = list(texts)  # pass-through default
        times_ms = np.zeros(n)
        translated = np.zeros(n, dtype=bool)

        # Group indices by source language so each batch is monolingual.
        groups: dict[str, list[int]] = defaultdict(list)
        skipped_langs: set[str] = set()
        for idx, lang in enumerate(languages):
            if lang == "en":
                continue
            if lang not in FLORES_CODES:
                skipped_langs.add(lang)
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
                if self.cfg.family == "nllb":
                    self.tokenizer.src_lang = src_flores
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

                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    t0 = time.perf_counter()
                    generated = self.model.generate(**encoded, **self._generate_kwargs())
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    decoded = self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    for i, translation in zip(index_batch, self._postprocess(decoded)):
                        out_texts[i] = translation if translation else texts[i]
                        times_ms[i] = elapsed_ms / len(index_batch)
                        translated[i] = True
                    progress.update(len(index_batch))
        progress.close()

        total_time = time.perf_counter() - total_start
        logger.info(
            "%s translated %d/%d posts in %.1fs",
            self.cfg.display_name, int(translated.sum()), n, total_time,
        )
        return TranslationResult(
            texts=out_texts,
            times_ms=times_ms,
            total_time_s=total_time,
            gpu_peak_mb=get_gpu_peak_mb(self.device),
            translated_mask=translated,
        )

    def free(self) -> None:
        del self.model
        del self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
