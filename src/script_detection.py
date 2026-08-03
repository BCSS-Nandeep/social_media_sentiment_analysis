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


def detect_script(text: str) -> str:
    """Return the dominant script in *text*: a key of UNICODE_RANGES, 'latin',
    or 'unknown' if no alphabetic character matched any known range."""
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if LATIN_RANGE[0] <= cp <= LATIN_RANGE[1]:
            counts["latin"] += 1
            continue
        for name, (lo, hi) in UNICODE_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def is_latin_script(text: str) -> bool:
    return detect_script(text) == "latin"
