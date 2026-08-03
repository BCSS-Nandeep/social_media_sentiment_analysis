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
flagged as Latin-script — native-script detection (lingua) is untouched.
"""
from __future__ import annotations

import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import config

logger = logging.getLogger("benchmark.lid_roman")

FTR_ZIP_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-ftr.zip"
FTR_MODEL_FILE = config.MODELS_DIR / "indiclid-ftr" / "model_baseline_roman.bin"

BERT_ZIP_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/indiclid-bert.zip"
BERT_MODEL_FILE = config.MODELS_DIR / "indiclid-bert" / "basline_nn_simple.pt"
BERT_TOKENIZER = "ai4bharat/IndicBERTv2-MLM-only"

FTR_CONFIDENCE_THRESHOLD = 0.6  # same default AI4Bharat's own IndicLID class uses

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
    "tel_Latn": "te", "urd_Latn": "ur", "eng_Latn": "en",
}


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

    def _ftr_predict(self, text: str) -> tuple[str, float] | None:
        if self.ftr is None:
            return None
        labels, scores = self.ftr.predict(text.replace("\n", " "))
        return labels[0].replace("__label__", ""), float(scores[0])

    def _bert_predict(self, text: str) -> str | None:
        import torch

        if self.bert is None or self.tokenizer is None:
            return None
        encoded = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.bert(**encoded)
        idx = int(out.logits.argmax(dim=1)[0])
        return LABELS_BY_INDEX[idx] if idx < len(LABELS_BY_INDEX) else None

    def detect(self, text: str) -> str | None:
        """Best-effort language for Latin-script `text`. None means "couldn't
        improve on the existing guess" — caller keeps what it already had."""
        if not text.strip():
            return None
        try:
            ftr_result = self._ftr_predict(text)
            if ftr_result is not None:
                label, score = ftr_result
                if score >= FTR_CONFIDENCE_THRESHOLD:
                    return LABEL_TO_LANG.get(label)
            # FTR wasn't confident (or unavailable) — try the BERT reranker.
            bert_label = self._bert_predict(text)
            if bert_label is not None:
                return LABEL_TO_LANG.get(bert_label)
            return None
        except Exception as exc:
            logger.warning("Roman LID failed for text: %s", exc)
            return None
