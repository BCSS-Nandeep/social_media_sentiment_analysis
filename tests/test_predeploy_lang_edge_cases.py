"""Pre-deploy smoke: every DETECTOR language × common social edge cases.

Does not load IndicTrans2/Cardiff. Checks clean/LID/script/API limits and that
sarcasm still surfaces review_recommended without rewriting sentiment.
"""
from __future__ import annotations

import json
import unittest
from collections import defaultdict
from pathlib import Path

import config
from src.language_detector import LanguageDetector
from src.pipeline import PipelineResult, SentimentPipeline, annotate_result_quality
from src.preprocessing import clean_text
from src.script_detection import detect_script, unique_indic_language
from src.transliteration import should_preserve_roman

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "predeploy_lang_edge_cases.json"

# Edge families we expect at least once in the fixture (not every lang needs all).
REQUIRED_EDGE_CASES = {
    "native_positive",
    "native_negative",
    "emoji_hashtag",
    "url_mention",
    "short_sparse",
    "mixed_script",
    "romanized_code_mix",
    "sarcasm",
}


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class PredeployLangEdgeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load()
        cls.posts = cls.payload["posts"]
        cls.detector = LanguageDetector(seed=config.SEED)
        if len(cls.posts) < 40:
            raise AssertionError("lang edge-case fixture too small")

    def test_fixture_covers_every_detector_language(self):
        covered = {p["lang_target"] for p in self.posts}
        missing = set(config.DETECTOR_LANGUAGES) - covered
        self.assertFalse(missing, f"missing languages: {sorted(missing)}")

    def test_fixture_covers_required_edge_families(self):
        present = {p["edge_case"] for p in self.posts}
        missing = REQUIRED_EDGE_CASES - present
        self.assertFalse(missing, f"missing edge cases: {sorted(missing)}")

    def test_each_language_has_multiple_edge_cases(self):
        by_lang: dict[str, set[str]] = defaultdict(set)
        for post in self.posts:
            by_lang[post["lang_target"]].add(post["edge_case"])
        for lang in config.DETECTOR_LANGUAGES:
            with self.subTest(lang=lang):
                self.assertGreaterEqual(
                    len(by_lang[lang]),
                    5,
                    f"{lang} only has {sorted(by_lang[lang])}",
                )

    def test_posts_fit_api_request_limits(self):
        for post in self.posts:
            self.assertLessEqual(
                len(post["post_text"]),
                config.API_MAX_TEXT_CHARS,
                post["id"],
            )
        self.assertLessEqual(len(self.posts), config.API_MAX_TEXTS)
        total = sum(len(p["post_text"]) for p in self.posts)
        self.assertLessEqual(total, config.API_MAX_TOTAL_CHARS)

    def test_clean_never_crashes_and_url_mentions_are_stripped(self):
        for post in self.posts:
            with self.subTest(post["id"]):
                cleaned = clean_text(post["post_text"])
                if post["edge_case"] == "url_mention":
                    self.assertNotIn("http", cleaned.lower())
                    self.assertNotIn("@", cleaned)
                if post["edge_case"] != "short_sparse":
                    self.assertTrue(cleaned.strip(), msg=post["id"])

    def test_emoji_hashtag_posts_keep_cues(self):
        emoji_posts = [p for p in self.posts if p["edge_case"] == "emoji_hashtag"]
        self.assertGreaterEqual(len(emoji_posts), len(config.DETECTOR_LANGUAGES) - 1)
        for post in emoji_posts:
            cleaned = clean_text(post["post_text"])
            with self.subTest(post["id"]):
                self.assertTrue(
                    any(ch in cleaned for ch in "🔥😡👏🙄"),
                    msg=f"emoji lost in {post['id']}: {cleaned!r}",
                )

    def test_script_matches_expected_when_present(self):
        for post in self.posts:
            expected = post.get("expected_script")
            if not expected:
                continue
            script = detect_script(post["post_text"])
            with self.subTest(post["id"], script=script):
                self.assertIn(script, expected)

    def test_language_detector_within_expected_band(self):
        allowed = set(config.DETECTOR_LANGUAGES) | {"unknown"}
        for post in self.posts:
            cleaned = clean_text(post["post_text"])
            lang = self.detector.detect(cleaned)
            with self.subTest(post["id"], lang=lang):
                self.assertIn(lang, allowed)
                if self.detector.backend == "none":
                    continue
                expected = post.get("expected_lang")
                if expected:
                    self.assertIn(lang, expected, msg=post["id"])

    def test_native_positive_negative_not_latin_for_indic_scripts(self):
        """Native-script Indic posts must not be classified as pure Latin."""
        non_latin_langs = {
            "hi", "te", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur",
        }
        for post in self.posts:
            if post["lang_target"] not in non_latin_langs:
                continue
            if post["edge_case"] not in {"native_positive", "native_negative"}:
                continue
            with self.subTest(post["id"]):
                self.assertNotEqual("latin", detect_script(post["post_text"]))

    def test_kannada_malayalam_script_fallback_routes_to_supported_lang(self):
        """lingua cannot load KN/ML; pipeline script fallback must still route."""
        service = SentimentPipeline.__new__(SentimentPipeline)

        class FakeRoman:
            def detect(self, text):
                return "en"

        service.roman_detector = FakeRoman()
        for post in self.posts:
            if post["lang_target"] not in {"kn", "ml"}:
                continue
            if post["edge_case"] not in {
                "native_positive", "native_negative", "mixed_script",
            }:
                continue
            with self.subTest(post["id"]):
                self.assertEqual(post["lang_target"], unique_indic_language(post["post_text"]))
                refined = service._refine_latin_languages(
                    [post["post_text"]],
                    ["unknown" if post["edge_case"].startswith("native") else "en"],
                )
                self.assertEqual([post["lang_target"]], refined)

    def test_short_tenglish_has_romanized_cues_english_does_not(self):
        from src.lid_roman import romanized_indic_cue_count

        tenglish = next(p for p in self.posts if p["id"] == "te-romanized_code_mix")
        english = next(p for p in self.posts if p["id"] == "en-native_positive")
        self.assertGreaterEqual(romanized_indic_cue_count(tenglish["post_text"]), 1)
        self.assertEqual(0, romanized_indic_cue_count(english["post_text"]))

    def test_mixed_script_posts_contain_latin_and_indic(self):
        """Mixed posts must include both Latin and a native Indic block."""
        indic_blocks = {
            "devanagari", "bengali", "gurmukhi", "gujarati",
            "tamil", "telugu", "kannada", "malayalam", "perso_arabic",
        }
        for post in self.posts:
            if post["edge_case"] != "mixed_script":
                continue
            if post["lang_target"] == "en":
                continue
            text = post["post_text"]
            with self.subTest(post["id"]):
                self.assertTrue(any("a" <= ch.lower() <= "z" for ch in text))
                self.assertTrue(
                    any(
                        ord(ch) >= 0x0600 and detect_script(ch) in indic_blocks
                        for ch in text
                        if not ch.isspace()
                    )
                    or detect_script(text) in indic_blocks
                    or any(detect_script(ch) in indic_blocks for ch in text if ch.strip()),
                    msg=f"no Indic script chars in {post['id']}",
                )

    def test_english_tokens_preserved_for_xlit_path(self):
        self.assertTrue(should_preserve_roman("Good"))
        self.assertTrue(should_preserve_roman("BCCI"))
        self.assertTrue(should_preserve_roman("ICC"))
        self.assertTrue(should_preserve_roman("Dont") or should_preserve_roman("not"))
        self.assertTrue(should_preserve_roman("the"))
        self.assertTrue(should_preserve_roman("and"))

    def test_sarcasm_posts_flag_review_without_flipping_label(self):
        sarcasm = [p for p in self.posts if p["edge_case"] == "sarcasm"]
        self.assertGreaterEqual(len(sarcasm), 8)
        for post in sarcasm[:5]:
            result = PipelineResult(
                post_text=post["post_text"],
                language=post["lang_target"],
                english_text=clean_text(post["post_text"]),
                was_translated=post["lang_target"] != "en",
                was_transliterated=False,
                sentiment="Positive",
                confidence=0.42,
                translation_time_ms=0.0,
                sentiment_time_ms=1.0,
                total_time_ms=1.0,
            )
            flagged = annotate_result_quality(result)
            with self.subTest(post["id"]):
                self.assertEqual("Positive", flagged.sentiment)
                self.assertTrue(flagged.review_recommended)


if __name__ == "__main__":
    unittest.main()
