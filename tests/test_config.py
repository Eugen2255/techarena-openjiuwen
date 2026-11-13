"""Тесты настроек (без openJiuwen). Запуск: python -m unittest discover -s tests"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.config import DEFAULT_BUDGET, parse_budget, parse_settings  # noqa: E402


class BudgetTest(unittest.TestCase):
    def test_formats(self):
        # Организаторы передадут бюджет в BUDGET и предупредили, что он может отличаться от 100K.
        for text, expected in {"100000": 100000, "100000.0": 100000, "1e5": 100000, "100_000": 100000,
                               "100 000": 100000, "100,000": 100000, "100k": 100000, "100K": 100000,
                               "0.1M": 100000, " 50000 ": 50000, "75000 tokens": 75000}.items():
            with self.subTest(text=text):
                self.assertEqual(parse_budget(text), expected)

    def test_unreadable_is_none(self):
        for text in (None, "", "abc", "-5", "0", "one hundred"):
            with self.subTest(text=text):
                self.assertIsNone(parse_budget(text))

    def test_settings_take_budget_from_env(self):
        with mock.patch.dict(os.environ, {"BUDGET": "30k"}, clear=False):
            self.assertEqual(parse_settings([]).token_budget, 30000)
        with mock.patch.dict(os.environ, {"BUDGET": "garbage"}, clear=False):
            self.assertEqual(parse_settings([]).token_budget, DEFAULT_BUDGET)  # не падаем при старте
        env = {k: v for k, v in os.environ.items() if k not in ("BUDGET", "HARNESS_TOKEN_BUDGET")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(parse_settings([]).token_budget, DEFAULT_BUDGET)


if __name__ == "__main__":
    unittest.main()
