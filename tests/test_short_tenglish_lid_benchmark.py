"""Focused short Tenglish vs English routing benchmark (LID only).

Live IndicLID runs only when the FTR checkpoint is already on disk.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from src.lid_roman import (
    FTR_MODEL_FILE,
    romanized_indic_cue_count,
    unique_romanized_cue_language,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "short_tenglish_lid_benchmark.json"
)


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _route(lang: str | None) -> str:
    if lang is None or lang == "en":
        return "en"
    return "indic"


class ShortTenglishBenchmarkFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load()
        cls.posts = cls.payload["posts"]

    def test_fixture_counts(self):
        counts = Counter(p["split"] for p in self.posts)
        self.assertEqual(100, counts["short_tenglish"])
        self.assertEqual(100, counts["short_english"])
        self.assertEqual(50, counts["tenglish_english"])
        self.assertEqual(50, counts["ambiguous"])

    def test_ambiguous_has_no_gold_language(self):
        for post in self.posts:
            if post["split"] == "ambiguous":
                self.assertEqual("ambiguous", post["gold_route"])

    def test_english_split_has_no_romanized_cues(self):
        for post in self.posts:
            if post["split"] != "short_english":
                continue
            self.assertEqual(
                0,
                romanized_indic_cue_count(post["text"]),
                post["text"],
            )

    def test_repeated_vowel_cue_collapses(self):
        self.assertGreaterEqual(romanized_indic_cue_count("chaala baagundi"), 1)
        self.assertEqual("te", unique_romanized_cue_language("chaala baagundi"))


@unittest.skipUnless(FTR_MODEL_FILE.exists(), "IndicLID-FTR checkpoint not present")
class ShortTenglishLiveLidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.lid_roman import RomanLanguageDetector

        cls.posts = _load()["posts"]
        cls.detector = RomanLanguageDetector()

    def test_routing_metrics(self):
        te_n = en_n = te_hit = en_hit = en_fp = 0
        mix_n = mix_hit = 0
        for post in self.posts:
            pred = _route(self.detector.detect(post["text"]))
            gold = post["gold_route"]
            if gold == "ambiguous":
                continue
            if post["split"] == "short_tenglish":
                te_n += 1
                te_hit += pred == "indic"
            elif post["split"] == "short_english":
                en_n += 1
                en_hit += pred == "en"
                en_fp += pred == "indic"
            elif post["split"] == "tenglish_english":
                mix_n += 1
                mix_hit += pred == "indic"
        self.assertGreaterEqual(te_hit / te_n, 0.85)
        self.assertGreaterEqual(en_hit / en_n, 0.95)
        self.assertLessEqual(en_fp / en_n, 0.05)
        self.assertGreaterEqual(mix_hit / mix_n, 0.70)
