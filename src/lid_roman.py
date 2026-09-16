"""Roman-script language identification — AI4Bharat IndicLID (FTR + BERT rerank).

The general-purpose language detector (src/language_detector.py, lingua/
langdetect) is fine for native-script text but cannot tell romanized Indic
text apart from English: neither lingua nor fastText's generic lid.176 have
a "Hindi written in Latin letters" class, since they're trained on native-
script corpora. Tested empirically: 0/7 correct on common Hinglish phrases
with both.

IndicLID is AI4Bharat's purpose-built fix — a 2-stage ensemble:
  1. IndicLID-FTR (fastText) — fast, but tested to be *confidently wrong* on
     short colloquial phrases (e.g. 93-100% confidence for the wrong
     language), not just low-recall. A confidence threshold alone doesn't
     catch this, since wrong answers can outscore right ones.
  2. IndicLID-BERT — reranks whatever FTR wasn't confident about. Loaded via
     torch.load(weights_only=False): a full pickled object, not just
     weights — this deserializes as a plain transformers.BertForSequence
     Classification (confirmed by loading it), so no custom class needed,
     but it does mean trusting the pickle. Source: AI4Bharat's official
     IndicLID GitHub release (MIT license) — accepted per explicit sign-off,
     since arbitrary pickle deserialization is a real risk regardless of
     source reputation.

Even with both stages, tested accuracy on short phrases still confuses
closely-related languages (e.g. Hindi <-> Maithili/Urdu/Punjabi) — this
matches a limitation AI4Bharat's own paper explicitly documents, not a bug
in this integration. Treat this stage as "meaningfully better than generic
LID," not "reliable" — same graceful-fallback contract as translation and
transliteration: any failure or low confidence keeps the existing detector's
guess rather than forcing a possibly-wrong override.

Only used to REFINE the language for text that script_detection.py already
flagged as Latin-script — native-script Kannada/Malayalam (and other unique
scripts) are recovered in pipeline._refine_latin_languages via
unique_indic_language, because lingua has no KN/ML profiles.
"""
from __future__ import annotations

import logging
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import config
from src.inference_watchdog import guarded_model_call

logger = logging.getLogger("benchmark.lid_roman")

FTR_ZIP_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip"
FTR_MODEL_FILE = config.MODELS_DIR / "indiclid-ftr" / "model_baseline_roman.bin"

BERT_ZIP_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip"
BERT_MODEL_FILE = config.MODELS_DIR / "indiclid-bert" / "basline_nn_simple.pt"
BERT_TOKENIZER = "ai4bharat/IndicBERTv2-MLM-only"

FTR_CONFIDENCE_THRESHOLD = 0.6  # same default AI4Bharat's own IndicLID class uses
# When FTR is confident English, still consider a lower-ranked Indic class if
# the post also has romanized Indic function words (short Tenglish). From
# IndicLID paper: accuracy collapses below ~10 words and FTR is often
# confidently wrong on romanized vs English.
FTR_ALTERNATE_INDIC_MIN_SCORE = 0.15
# Residual (English-stripped) FTR is only consulted after full-text FTR/BERT
# already said English. 0.25 sits below the English-heavy Tenglish recovery
# score seen in probes (~0.33) and well above stray-token noise.
FTR_RESIDUAL_INDIC_MIN_SCORE = 0.25
FTR_TOPK = 3

# High-precision romanized function/content words, mapped to a language.
# Not a general dictionary: English-looking posts without these stay English.
# Sources: existing production cues; attested Telugu roman function words
# (chala/undi/nenu/…); Dravidian/Hindi counterparts already used in routing.
# Spelling variants are matched by collapsing repeated letters (chaala→chala).
_ROMANIZED_INDIC_CUE_LANG: dict[str, str] = {
    "chala": "te", "undi": "te", "nenu": "te", "chalu": "te", "bagunna": "te",
    "ayyindi": "te", "unnadi": "te", "cheyandi": "te", "ledu": "te",
    "undhi": "te", "unna": "te", "emi": "te", "enduku": "te", "ipudu": "te",
    "inka": "te", "kaani": "te", "kada": "te", "gaa": "te", "ani": "te",
    "bagundi": "te", "baagundi": "te", "chestunnav": "te", "nijam": "te",
    "enti": "te", "unnaru": "te", "vallu": "te", "ra": "te",
    "assalu": "te", "bagoledu": "te", "chesaru": "te", "kaadu": "te",
    "kadu": "te", "manchidi": "te", "cheyyali": "te",
    "thumba": "kn",
    "romba": "ta", "irukku": "ta",
    "aanu": "ml",
    "aahe": "mr",
    "kharab": "hi", "bakwas": "hi", "bilkul": "hi",
}
_ROMANIZED_INDIC_CUES = frozenset(_ROMANIZED_INDIC_CUE_LANG)
_CUE_REPEAT_RE = re.compile(r"(.)\1+")
_LATIN_SPLIT_RE = re.compile(r"[.!,?;:]+")

# Both FTR and BERT share this label space (BERT's classifier head has
# exactly 22 outputs: the *_Latn codes below + 'other' — verified by loading
# it and checking model.config.num_labels == 22).
LABELS_BY_INDEX = [
    "asm_Latn", "ben_Latn", "brx_Latn", "guj_Latn", "hin_Latn", "kan_Latn",
    "kas_Latn", "kok_Latn", "mai_Latn", "mal_Latn", "mni_Latn", "mar_Latn",
    "nep_Latn", "ori_Latn", "pan_Latn", "san_Latn", "snd_Latn", "tam_Latn",
    "tel_Latn", "urd_Latn", "eng_Latn", "other",
]
LABEL_TO_LANG: dict[str, str] = {
    "hin_Latn": "hi", "ben_Latn": "bn", "guj_Latn": "gu", "kan_Latn": "kn",
    "mal_Latn": "ml", "mar_Latn": "mr", "pan_Latn": "pa", "tam_Latn": "ta",
    "tel_Latn": "te", "urd_Latn": "ur", "eng_Latn": "en", "ori_Latn": "or",
}


def _cue_lookup_keys(core: str) -> list[str]:
    folded = core.casefold()
    collapsed = _CUE_REPEAT_RE.sub(r"\1", folded)
    if collapsed == folded:
        return [folded]
    return [folded, collapsed]


def _romanized_indic_cue_hits(text: str) -> list[tuple[str, str]]:
    from src.transliteration import ROMAN_WORD_RE, should_preserve_roman

    hits: list[tuple[str, str]] = []
    for token in _LATIN_SPLIT_RE.sub(" ", text).split():
        match = ROMAN_WORD_RE.fullmatch(token)
        if not match:
            continue
        core = match.group(2)
        if should_preserve_roman(core):
            continue
        for key in _cue_lookup_keys(core):
            lang = _ROMANIZED_INDIC_CUE_LANG.get(key)
            if lang:
                hits.append((key, lang))
                break
    return hits


def romanized_indic_cue_count(text: str) -> int:
    return len(_romanized_indic_cue_hits(text))


def unique_romanized_cue_language(text: str) -> str | None:
    langs = {lang for _, lang in _romanized_indic_cue_hits(text)}
    if len(langs) == 1:
        return langs.pop()
    return None


def latin_residual_without_preserved_english(text: str) -> str:
    """Latin tokens that IndicXlit would actually transliterate.

    Word-level EN–TE LID (Gundapu et al., arXiv:2010.04482) motivates
    stripping English tokens before a sentence-level LID decision.
    """
    from src.transliteration import ROMAN_WORD_RE, should_preserve_roman

    kept: list[str] = []
    for token in _LATIN_SPLIT_RE.sub(" ", text).split():
        match = ROMAN_WORD_RE.fullmatch(token)
        if not match:
            continue
        core = match.group(2)
        if should_preserve_roman(core) or len(core) <= 1:
            continue
        kept.append(core)
    return " ".join(kept)


def _indic_alternative_from_ftr(ranked: list[tuple[str, float]], text: str) -> str | None:
    """Prefer a lower-ranked Indic FTR class only when romanized cues exist."""
    if romanized_indic_cue_count(text) < 1:
        return None
    for label, score in ranked:
        if label == "eng_Latn" or score < FTR_ALTERNATE_INDIC_MIN_SCORE:
            continue
        mapped = LABEL_TO_LANG.get(label)
        if mapped and mapped != "en":
            return mapped
    return None


def _download_zip(url: str, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "model.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_to)


def _patch_fasttext_numpy() -> None:
    """fasttext 0.9.2's predict() calls np.array(probs, copy=False), which
    numpy>=2.0 rejects outright (it now raises instead of silently copying).
    Patch just the copy kwarg away; harmless no-op once fasttext catches up."""
    import numpy as np

    if getattr(np.array, "_lid_roman_patched", False):
        return
    original = np.array

    def patched(*args, **kwargs):
        kwargs.pop("copy", None)
        return original(*args, **kwargs)

    patched._lid_roman_patched = True
    np.array = patched


class RomanLanguageDetector:
    """Load once, then call :meth:`detect`. Returns None (caller keeps the
    existing detector's guess) on any failure or low-confidence prediction."""

    def __init__(self, device=None) -> None:
        import torch

        self.device = device or torch.device("cpu")
        self.ftr = None
        self.bert = None
        self.tokenizer = None

        try:
            if not FTR_MODEL_FILE.exists():
                logger.info("Downloading IndicLID-FTR (one-time, ~280MB)...")
                _download_zip(FTR_ZIP_URL, FTR_MODEL_FILE.parent.parent)
            _patch_fasttext_numpy()
            import fasttext

            fasttext.FastText.eprint = lambda *_: None  # silence the load-time warning
            self.ftr = fasttext.load_model(str(FTR_MODEL_FILE))
            logger.info("IndicLID-FTR ready.")
        except Exception as exc:
            logger.error("IndicLID-FTR setup failed — Latin-script text keeps the "
                         "general detector's guess: %s", exc)

        try:
            if not BERT_MODEL_FILE.exists():
                logger.info("Downloading IndicLID-BERT (one-time, ~1.1GB)...")
                _download_zip(BERT_ZIP_URL, BERT_MODEL_FILE.parent.parent)
            from transformers import AutoTokenizer

            self.bert = torch.load(str(BERT_MODEL_FILE), map_location=self.device, weights_only=False)
            self.bert.eval()
            if self.device.type == "cuda":
                self.bert.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER)
            logger.info("IndicLID-BERT reranker ready.")
        except Exception as exc:
            logger.error("IndicLID-BERT setup failed — falling back to FTR-only "
                         "(less reliable on short/ambiguous phrases): %s", exc)
            self.bert = None
            self.tokenizer = None

    def _ftr_topk(self, text: str) -> list[tuple[str, float]]:
        if self.ftr is None:
            return []
        labels, scores = self.ftr.predict(text.replace("\n", " "), k=FTR_TOPK)
        return [
            (str(label).replace("__label__", ""), float(score))
            for label, score in zip(labels, scores)
        ]

    def _ftr_predict(self, text: str) -> tuple[str, float] | None:
        top = self._ftr_topk(text)
        if not top:
            return None
        return top[0]

    def _bert_predict(self, text: str) -> str | None:
        import torch

        if self.bert is None or self.tokenizer is None:
            return None
        encoded = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)

        def predict_label():
            encoded_on_device = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                out = self.bert(**encoded_on_device)
            idx = int(out.logits.argmax(dim=1)[0])
            return LABELS_BY_INDEX[idx] if idx < len(LABELS_BY_INDEX) else None

        return guarded_model_call(
            predict_label, config.INFERENCE_HARD_TIMEOUT_S
        )

    def _rescue_romanized_indic(
        self, text: str, ranked: list[tuple[str, float]]
    ) -> str | None:
        """Secondary check used only after sentence-level LID said English."""
        alternate = _indic_alternative_from_ftr(ranked, text)
        if alternate:
            return alternate
        if romanized_indic_cue_count(text) < 1:
            return None
        residual = latin_residual_without_preserved_english(text)
        if not residual.strip():
            return None
        residual_ranked = self._ftr_topk(residual)
        if not residual_ranked:
            return None
        rlabel, rscore = residual_ranked[0]
        if rscore < FTR_RESIDUAL_INDIC_MIN_SCORE or rlabel == "eng_Latn":
            return None
        cue_lang = unique_romanized_cue_language(text)
        mapped = LABEL_TO_LANG.get(rlabel)
        if mapped and mapped != "en":
            return cue_lang or mapped
        # Unmapped roman classes (ori/snd/mni/…) still mean "not English".
        return cue_lang

    def detect(self, text: str) -> str | None:
        """Best-effort language for Latin-script `text`. None means "couldn't
        improve on the existing guess" — caller keeps what it already had."""
        if not text.strip():
            return None
        try:
            ranked = self._ftr_topk(text)
            ftr_result = ranked[0] if ranked else None
            cues = romanized_indic_cue_count(text)
            residual_tokens = latin_residual_without_preserved_english(text).split()
            short_no_cue = cues == 0 and len(residual_tokens) <= 4
            confident_ftr_label = None
            if ftr_result is not None:
                label, score = ftr_result
                if score >= FTR_CONFIDENCE_THRESHOLD:
                    confident_ftr_label = label
                    mapped = LABEL_TO_LANG.get(label)
                    # Short Latin with no romanized cues is usually English;
                    # FTR/BERT often assign a random Indic class (pa/bn/gu).
                    if mapped and mapped != "en" and not short_no_cue:
                        return unique_romanized_cue_language(text) or mapped
            bert_label = self._bert_predict(text)
            if bert_label is not None:
                mapped = LABEL_TO_LANG.get(bert_label)
                if mapped and mapped != "en" and not short_no_cue:
                    return unique_romanized_cue_language(text) or mapped
                if mapped == "en" or short_no_cue:
                    return self._rescue_romanized_indic(text, ranked) or "en"
            if confident_ftr_label is not None:
                mapped = LABEL_TO_LANG.get(confident_ftr_label)
                if mapped == "en" or mapped is None:
                    rescued = self._rescue_romanized_indic(text, ranked)
                    if rescued:
                        return rescued
                    return "en" if mapped == "en" else rescued
                if short_no_cue:
                    return "en"
                return mapped
            return self._rescue_romanized_indic(text, ranked)
        except Exception as exc:
            logger.warning("Roman LID failed for text: %s", exc)
            return None

    def free(self) -> None:
        """Release the FTR and BERT models. Idempotent.

        IndicLID-BERT is the largest artifact the pipeline loads (~1.1 GB), so
        this matters on shutdown even though the stage itself is optional.
        """
        import torch

        self.ftr = None
        self.bert = None
        self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
