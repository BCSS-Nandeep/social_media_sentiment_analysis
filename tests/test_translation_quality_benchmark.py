"""Translation-quality fixture checks (no GPU / no Cardiff)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.lid_roman import romanized_indic_cue_count
from src.transliteration import TELUGU_NATIVE_OVERRIDES, should_preserve_roman
from src.translation import split_translation_chunks, translation_is_usable

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "translation_quality_benchmark.json"


class TranslationQualityFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.posts = json.loads(FIXTURE.read_text(encoding="utf-8"))["posts"]

    def test_fixture_marks_critical_terms(self):
        self.assertGreaterEqual(len(self.posts), 10)
        for post in self.posts:
            self.assertTrue(post["source_text"].strip())
            self.assertTrue(post["critical_phrase"])
            self.assertTrue(post["important_terms"])

    def test_tenglish_sentiment_stems_are_overridden(self):
        for stem in ("bagundi", "kaadu", "bagoledu", "assalu", "undi"):
            self.assertIn(stem, TELUGU_NATIVE_OVERRIDES)

    def test_code_mix_english_is_preserved(self):
        self.assertTrue(should_preserve_roman("use"))
        self.assertTrue(should_preserve_roman("improve"))
        self.assertTrue(should_preserve_roman("better"))
        self.assertTrue(should_preserve_roman("worst"))

    def test_new_tenglish_cues_are_present(self):
        self.assertGreaterEqual(romanized_indic_cue_count("assalu bagoledu"), 1)
        self.assertGreaterEqual(romanized_indic_cue_count("super ga chesaru"), 1)
        self.assertGreaterEqual(romanized_indic_cue_count("bagundi kaadu"), 1)

    def test_english_bypass_case_has_no_indic_cues(self):
        english = next(p for p in self.posts if p["id"] == "en-bypass")
        self.assertEqual(0, romanized_indic_cue_count(english["source_text"]))

    def test_sentence_split_is_opt_in_for_multi_sentence_only(self):
        self.assertEqual(1, len(split_translation_chunks("chala bagundi")))
        self.assertGreaterEqual(
            len(split_translation_chunks("One. Two. Three.")),
            3,
        )

    def test_translator_still_imports_batching_helpers(self):
        import src.translation as mod

        self.assertTrue(callable(mod.chunked))
        self.assertTrue(callable(mod.reset_gpu_peak))

    def test_usable_gate_preserves_polarity_class_not_exact_token(self):
        # Polarity loss — no sentiment left
        self.assertFalse(
            translation_is_usable("chala worst ga undi", "It continues")
        )
        # Exact token still accepted
        self.assertTrue(
            translation_is_usable("chala worst ga undi", "It is the worst")
        )
        # Synonym / paraphrase must be accepted (previous lexical gate failed these)
        self.assertTrue(
            translation_is_usable("chala worst ga undi", "It's very bad")
        )
        self.assertTrue(
            translation_is_usable("good ga chesaru", "well done")
        )
        # Polarity reversal must still be rejected
        self.assertFalse(
            translation_is_usable("chala worst ga undi", "It's very good")
        )
        self.assertFalse(
            translation_is_usable("good ga chesaru", "It was terrible")
        )
