"""Romanized-Indic transliteration — AI4Bharat IndicXlit (Roman -> native script).

Loads the raw IndicXlit fairseq checkpoint directly via `fairseq` (not the
`ai4bharat-transliteration` PyPI wrapper): that wrapper unconditionally
imports `urduhack`, which pulls in TensorFlow, which segfaults in this
process once torch/transformers (IndicTrans2, Cardiff) are already loaded —
confirmed by direct testing (see repo notes). Loading the same weights
through plain `fairseq.models.transformer.TransformerModel` avoids +
TensorFlow entirely and works cleanly alongside the rest of the pipeline.

IndicXlit works at the CHARACTER level (dict.en.txt has 54 symbol types) and
is multilingual: one checkpoint, target language selected via task config at
generation time. This module downloads the public checkpoint once (GitHub
release, no auth needed — unlike IndicTrans2's gated HF repo) and caches it
under MODELS_DIR.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch

import config
from src.inference_watchdog import guarded_model_call

logger = logging.getLogger("benchmark.transliteration")

MODEL_ZIP_URL = "https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-en-indic-v1.0.zip"
XLIT_DIR = config.MODELS_DIR / "indicxlit"

# The release zip doesn't include this. It's NOT the same list published at
# inference/python/lang_list.txt in the repo (that one is missing brx/mni,
# and using it raises "Error(s) in loading state_dict" — wrong embedding
# size for this specific v1.0 checkpoint). Order matters: this is the exact
# ordering the v1.0 checkpoint's multilingual vocabulary was built with,
# reconstructed from the checkpoint's own saved `lang_pairs` config and
# verified by loading successfully against it.
CHECKPOINT_LANG_LIST = (
    "en", "as", "bn", "brx", "gom", "gu", "hi", "kn", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sd", "si", "ta", "te", "ur",
)

# ponytail: one TransformerModel instance per target language (~11M params
# each) is simpler than sharing one model + rebuilding the generator per
# call, at the cost of ~10x the memory/startup time for a model this small.
# Revisit if startup time or memory becomes a real constraint.
SUPPORTED_LANGS = ("hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur")

ROMAN_WORD_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
URDU_NATIVE_OVERRIDES = {
    "teen": "تین",
    "din": "دن",
    "meh": "میں",
}
URDU_PRESERVE_ROMAN = {
    "mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday",
    "thu", "thur", "thurs", "thursday", "fri", "friday", "sat", "saturday",
    "sun", "sunday",
}


def _ensure_downloaded() -> None:
    if (XLIT_DIR / "corpus-bin" / "lang_list.txt").exists():
        return
    logger.info("Downloading IndicXlit checkpoint (one-time, ~130MB) from %s", MODEL_ZIP_URL)
    XLIT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "indicxlit.zip"
        urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # The release zip has corpus-bin/ and transformer/ at its root — no
        # wrapping folder.
        extracted = Path(tmp)
        shutil.copytree(extracted / "transformer", XLIT_DIR / "transformer")
        shutil.copytree(extracted / "corpus-bin", XLIT_DIR / "corpus-bin")
        (XLIT_DIR / "corpus-bin" / "lang_list.txt").write_text(
            "\n".join(CHECKPOINT_LANG_LIST) + "\n"
        )
    logger.info("IndicXlit checkpoint cached at %s", XLIT_DIR)


def _patch_multilingual_task() -> None:
    """fairseq-fixed's hub_utils passes `prefix_allowed_tokens_fn` to
    build_generator(), but TranslationMultiSimpleEpochTask.build_generator()
    doesn't accept it (fairseq API drift between the generic hub helper and
    this task subclass). Drop the unsupported kwarg; harmless no-op if a
    future fairseq version already supports it."""
    from fairseq.tasks.translation_multi_simple_epoch import (
        TranslationMultiSimpleEpochTask,
    )

    if getattr(TranslationMultiSimpleEpochTask, "_xlit_patched", False):
        return
    original = TranslationMultiSimpleEpochTask.build_generator

    def patched(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, **_ignored):
        return original(self, models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs)

    TranslationMultiSimpleEpochTask.build_generator = patched
    TranslationMultiSimpleEpochTask._xlit_patched = True


def _char_space(word: str) -> str:
    return " ".join(list(word))


def _prepare_word(word: str, lang: str) -> str:
    """Match IndicXlit's official Roman-input preprocessing."""
    return f"__{lang}__ {_char_space(word.lower())}"


class Transliterator:
    """Roman -> native-script transliteration for one or more Indic languages.

    Loads once, then call :meth:`transliterate`. Any failure during setup or
    per-call is caught and logged; callers should treat ``None`` as "could
    not transliterate — fall back to the original text" per the pipeline's
    graceful-degradation requirement.
    """

    def __init__(self, langs: tuple[str, ...] = SUPPORTED_LANGS, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.models: dict[str, object] = {}
        try:
            _ensure_downloaded()
            _patch_multilingual_task()
            import argparse

            torch.serialization.add_safe_globals([argparse.Namespace])
            from fairseq.models.transformer import TransformerModel

            lang_dict = str(XLIT_DIR / "corpus-bin" / "lang_list.txt")
            for lang in langs:
                try:
                    model = TransformerModel.from_pretrained(
                        model_name_or_path=str(XLIT_DIR / "transformer"),
                        checkpoint_file="indicxlit.pt",
                        data_name_or_path=str(XLIT_DIR / "corpus-bin"),
                        lang_dict=lang_dict,
                        lang_list=lang_dict,
                        source_lang="en",
                        target_lang=lang,
                    )
                    model.eval()
                    if self.device.type == "cuda":
                        model.cuda()
                    self.models[lang] = model
                except Exception as exc:
                    logger.warning("IndicXlit: failed to load target language %r: %s", lang, exc)
            logger.info("IndicXlit ready for %d languages: %s", len(self.models), sorted(self.models))
        except Exception as exc:
            logger.error("IndicXlit setup failed — transliteration disabled, pipeline will fall back: %s", exc)
            self.models = {}

    def supports(self, lang: str) -> bool:
        return lang in self.models

    def transliterate(self, text: str, lang: str) -> str | None:
        """Roman `text` -> native script for `lang`, preserving word breaks.
        Returns None (caller falls back to original text) on any failure."""
        model = self.models.get(lang)
        if model is None or not text.strip():
            return None
        try:
            words = text.split(" ")
            # One word at a time in a Python loop meant one fairseq generate()
            # call per word — each with its own batch-sampler setup overhead.
            # A 40-word post took 60+ seconds. fairseq's translate() batches a
            # list in a single call, so send every real word through at once.
            out_words = list(words)
            xlit_indices = []
            xlit_parts = []
            for i, word in enumerate(words):
                match = ROMAN_WORD_RE.fullmatch(word)
                if match is None:
                    continue
                prefix, core, suffix = match.groups()
                if any(char.isalnum() for char in prefix + suffix):
                    continue
                normalized = core.lower()
                if lang == "ur" and normalized in URDU_NATIVE_OVERRIDES:
                    out_words[i] = (
                        prefix + URDU_NATIVE_OVERRIDES[normalized] + suffix
                    )
                    continue
                if lang == "ur" and normalized in URDU_PRESERVE_ROMAN:
                    continue
                xlit_indices.append(i)
                xlit_parts.append((prefix, core, suffix))
            if not xlit_indices:
                return " ".join(out_words)
            spaced_batch = [
                _prepare_word(core, lang) for _prefix, core, _suffix in xlit_parts
            ]
            results = guarded_model_call(
                lambda: model.translate(spaced_batch, beam=5),
                config.INFERENCE_HARD_TIMEOUT_S,
            )
            for i, (prefix, _core, suffix), result in zip(
                xlit_indices, xlit_parts, results
            ):
                out_words[i] = prefix + result.replace(" ", "") + suffix
            return " ".join(out_words)
        except Exception as exc:
            logger.warning("IndicXlit: transliteration failed for lang=%s: %s", lang, exc)
            return None

    def free(self) -> None:
        self.models = {}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
