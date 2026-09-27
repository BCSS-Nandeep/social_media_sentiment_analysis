"""Regression check: unknown-language fallback must not reject unchanged
English output as an unusable translation (it never needed translating).
"""
from __future__ import annotations

import unittest

from src.translation import translation_is_usable


class UnknownLanguagePassthroughTest(unittest.TestCase):
    def test_unchanged_english_is_usable_when_language_unknown(self):
        text = "great work by the team today"
        self.assertTrue(
            translation_is_usable(text, text, require_change=False)
        )

    def test_unchanged_text_still_rejected_for_known_languages(self):
        text = "koi translation nahi hua"
        self.assertFalse(
            translation_is_usable(text, text, require_change=True)
        )

    def test_default_still_requires_change(self):
        text = "same text either way"
        self.assertFalse(translation_is_usable(text, text))

    def test_non_english_output_still_rejected_when_unknown(self):
        self.assertFalse(
            translation_is_usable("hi", "こんにちは", require_change=False)
        )


if __name__ == "__main__":
    unittest.main()
