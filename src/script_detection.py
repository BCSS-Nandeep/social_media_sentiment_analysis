"""Unicode-range script detection — no ML model, just character code points.

Used to tell native-script Indic text ("మోదీ ప్రభుత్వం") apart from the same
language typed in Latin letters ("Modi ప్రభుత్వం" -> "Modi prabhutvam"), so the
pipeline knows whether a post needs transliteration (IndicXlit) before
translation (IndicTrans2), or is already in the script IndicTrans2 expects.
"""
from __future__ import annotations

from collections import Counter

# (first_codepoint, last_codepoint) per script block actually used by the
# languages this pipeline supports (config.FLORES_CODES).
UNICODE_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Marathi
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),  # Punjabi
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "perso_arabic": (0x0600, 0x06FF),  # Urdu
}

LATIN_RANGE = (0x0041, 0x024F)  # basic Latin + Latin-1 + Latin Extended-A/B

# Scripts that uniquely identify a supported FLORES language. Devanagari is
# omitted because Hindi and Marathi share it.
UNIQUE_SCRIPT_TO_LANG: dict[str, str] = {
    "kannada": "kn",
    "malayalam": "ml",
    "tamil": "ta",
    "telugu": "te",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "bengali": "bn",
    "perso_arabic": "ur",
}


def _script_bucket(ch: str) -> str | None:
    """Map one character onto latin / an Indic block, including combining marks."""
    cp = ord(ch)
    for name, (lo, hi) in UNICODE_RANGES.items():
        if lo <= cp <= hi:
            return name
    if LATIN_RANGE[0] <= cp <= LATIN_RANGE[1] and ch.isalpha():
        return "latin"
    return None


def detect_script(text: str) -> str:
    """Return the dominant script in *text*: a key of UNICODE_RANGES, 'latin',
    or 'unknown' if no alphabetic character matched any known range."""
    counts: Counter[str] = Counter()
    for ch in text:
        bucket = _script_bucket(ch)
        if bucket is not None:
            counts[bucket] += 1
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def latin_letter_ratio(text: str) -> float:
    """Fraction of script-bearing characters that are Latin."""
    latin = other = 0
    for ch in text:
        bucket = _script_bucket(ch)
        if bucket is None:
            continue
        if bucket == "latin":
            latin += 1
        else:
            other += 1
    total = latin + other
    if total == 0:
        return 0.0
    return latin / total


def extract_latin_text(text: str) -> str:
    """Keep Latin letters, digits, and whitespace; drop Indic-script characters."""
    kept: list[str] = []
    for ch in text:
        bucket = _script_bucket(ch)
        if bucket == "latin" or ch.isspace() or (ch.isascii() and not ch.isalpha()):
            kept.append(ch)
    return " ".join("".join(kept).split())


def is_latin_script(text: str) -> bool:
    return detect_script(text) == "latin"


def unique_indic_language(text: str, min_share: float | None = None) -> str | None:
    """Return a supported language when one unique Indic script is present.

    Used when lingua cannot represent Kannada/Malayalam (no ISO 639-1 KN/ML
    profiles) and when mixed-script posts are mislabelled English because the
    Latin span dominates lingua. Devanagari is never mapped here.
    """
    import config

    threshold = config.INTELLIGENCE_CODEMIX_RATIO if min_share is None else min_share
    counts: Counter[str] = Counter()
    for ch in text:
        bucket = _script_bucket(ch)
        if bucket is not None:
            counts[bucket] += 1
    if not counts:
        return None
    total = sum(counts.values())
    indic = {
        name: n for name, n in counts.items()
        if name in UNIQUE_SCRIPT_TO_LANG
    }
    if not indic:
        return None
    script, n = max(indic.items(), key=lambda item: item[1])
    if n / total < threshold:
        return None
    return UNIQUE_SCRIPT_TO_LANG[script]
