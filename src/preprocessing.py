"""Text cleaning and dataset validation for social-media posts.

Cleaning steps (emojis are deliberately preserved — they carry sentiment):
    1. HTML entity unescaping and tag removal
    2. URL removal
    3. @mention removal
    4. Hashtag normalization (``#GoodWork`` -> ``GoodWork``)
    5. Unicode NFC normalization
    6. Whitespace collapsing
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata

import pandas as pd

from config import LABELS, REQUIRED_COLUMNS

logger = logging.getLogger("benchmark.preprocessing")

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_MULTISPACE_RE = re.compile(r"\s+")

_CANONICAL_LABELS = {label.lower(): label for label in LABELS}
# Common shorthand variants seen in real annotation files.
_CANONICAL_LABELS.update(
    {"pos": "Positive", "neg": "Negative", "neu": "Neutral", "neutral ": "Neutral"}
)


def clean_text(text: str) -> str:
    """Return the cleaned version of a single post (emojis preserved)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = unicodedata.normalize("NFC", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a required column is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def normalize_label(value: object) -> str:
    """Map a raw ground-truth value onto one of the canonical labels."""
    text = str(value).strip().lower()
    if text in _CANONICAL_LABELS:
        return _CANONICAL_LABELS[text]
    raise ValueError(
        f"Unrecognised ground-truth sentiment {value!r}; expected one of {LABELS}"
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize labels and clean every post.

    Adds a ``clean_text`` column and a normalized ``ground_truth`` column.
    Rows whose text is empty after cleaning are dropped (with a warning).
    """
    validate_columns(df)
    df = df.copy()
    df["ground_truth"] = df["ground_truth_sentiment"].map(normalize_label)
    df["clean_text"] = df["post_text"].map(clean_text)

    empty_mask = df["clean_text"].str.len() == 0
    if empty_mask.any():
        logger.warning(
            "Dropping %d row(s) whose post_text is empty after cleaning.",
            int(empty_mask.sum()),
        )
        df = df.loc[~empty_mask]

    df = df.reset_index(drop=True)
    logger.info("Preprocessing complete: %d posts ready for inference.", len(df))
    return df
