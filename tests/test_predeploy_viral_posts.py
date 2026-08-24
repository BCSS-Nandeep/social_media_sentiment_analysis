"""Pre-deploy smoke tests on Wigolo-collected viral public posts.

Does not load IndicTrans2/Cardiff (those need GPU + HF auth). It checks that
real noisy social text survives preprocessing, language routing, length limits,
and English-preserve transliteration without crashing.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import config
from src.language_detector import LanguageDetector
from src.pipeline import PipelineResult, annotate_result_quality
from src.preprocessing import clean_text
from src.script_detection import detect_script, latin_letter_ratio
from src.transliteration import ROMAN_WORD_RE, should_preserve_roman

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "predeploy_viral_posts.json"


def _load_posts() -> list[dict]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    posts = payload["posts"]
    if len(posts) < 8:
        raise AssertionError("predeploy fixture too small")
    return posts


class PredeployViralCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.posts = _load_posts()
        cls.detector = LanguageDetector(seed=config.SEED)

    def test_fixture_has_en_hi_and_te_scripts(self):
        texts = [p["post_text"] for p in self.posts]
        joined = "\n".join(texts)
        self.assertTrue(any("a" <= ch.lower() <= "z" for ch in joined))
        self.assertTrue(any("\u0900" <= ch <= "\u097f" for ch in joined))
        self.assertTrue(any("\u0c00" <= ch <= "\u0c7f" for ch in joined))

    def test_every_post_cleans_without_emptying_except_hashtag_only(self):
        for post in self.posts:
            with self.subTest(post["id"]):
                cleaned = clean_text(post["post_text"])
                if post["id"] == "url-1":
                    self.assertNotIn("http", cleaned)
                    self.assertTrue(cleaned)
                    continue
                self.assertTrue(cleaned.strip(), msg=post["id"])

    def test_posts_fit_api_request_limits(self):
        for post in self.posts:
            n = len(post["post_text"])
            self.assertLessEqual(n, config.API_MAX_TEXT_CHARS, post["id"])
        total = sum(len(p["post_text"]) for p in self.posts)
        self.assertLessEqual(total, config.API_MAX_TOTAL_CHARS)
        self.assertLessEqual(len(self.posts), config.API_MAX_TEXTS)

    def test_language_detector_returns_supported_or_unknown(self):
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

    def test_telugu_trailer_post_is_not_classified_latin_dominant(self):
        post = next(p for p in self.posts if p["id"] == "te-1")
        self.assertNotEqual("latin", detect_script(post["post_text"]))

    def test_tenglish_preserves_english_sentiment_words(self):
        post = next(p for p in self.posts if p["id"] == "tenglish-1")
        cores = []
        for token in post["post_text"].replace(".", " ").split():
            match = ROMAN_WORD_RE.fullmatch(token)
            if match:
                cores.append(match.group(2))
        self.assertIn("Good", cores)
        self.assertTrue(should_preserve_roman("Good"))
        self.assertTrue(should_preserve_roman("move"))

    def test_emoji_and_hashtag_posts_keep_sentiment_cues(self):
        police = next(p for p in self.posts if p["id"] == "trend-2")
        cleaned = clean_text(police["post_text"])
        self.assertIn("😡", cleaned)
        self.assertIn("UPPolice", cleaned)

    def test_sarcastic_meme_still_looks_review_worthy_at_low_confidence(self):
        meme = next(p for p in self.posts if p["id"] == "meme-2")
        result = PipelineResult(
            post_text=meme["post_text"],
            language="en",
            english_text=clean_text(meme["post_text"]),
            was_translated=False,
            was_transliterated=False,
            sentiment="Positive",
            confidence=0.41,
            translation_time_ms=0.0,
            sentiment_time_ms=1.0,
            total_time_ms=1.0,
        )
        flagged = annotate_result_quality(result)
        self.assertEqual("Positive", flagged.sentiment)
        self.assertTrue(flagged.review_recommended)

    def test_latin_ratio_on_code_mix_is_measurable(self):
        post = next(p for p in self.posts if p["id"] == "tenglish-1")
        ratio = latin_letter_ratio(post["post_text"])
        self.assertGreater(ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
