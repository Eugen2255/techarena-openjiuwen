"""Тесты учёта токенов (без LLM). Запуск: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.meter import BudgetExceeded, TokenMeter  # noqa: E402


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens, self.output_tokens, self.reasoning_tokens = input_tokens, output_tokens, 0


class TokenMeterTest(unittest.TestCase):
    def test_cost_is_output_plus_tenth_of_input(self):
        meter = TokenMeter(budget=1000)
        meter.record("t", FakeUsage(input_tokens=500, output_tokens=20))
        self.assertEqual(meter.total.cost, 70)
        self.assertEqual(meter.remaining(), 930)

    def test_failed_call_is_charged(self):
        # Запрос с ошибкой прокси тоже считает (встреча 09.11) — оценка входа и, при таймауте, потолок выхода.
        meter = TokenMeter(budget=1000)
        meter.charge("t", input_tokens=1200)
        meter.charge("t", input_tokens=300, output_tokens=400)
        self.assertEqual(meter.task_usage("t").calls, 2)
        self.assertEqual(meter.total.cost, 120 + 30 + 400)

    def test_reservation_blocks_budget_overrun(self):
        meter = TokenMeter(budget=1000)
        meter.reserve("a", 600)
        with self.assertRaises(BudgetExceeded):
            meter.reserve("b", 500)  # 600 уже зарезервировано идущим вызовом
        meter.release(600)
        meter.reserve("b", 500)

    def test_task_cap(self):
        meter = TokenMeter(budget=10000, task_cap=100)
        meter.charge("t", input_tokens=0, output_tokens=100)
        with self.assertRaises(BudgetExceeded):
            meter.reserve("t", 10)
        meter.reserve("other", 10)


if __name__ == "__main__":
    unittest.main()
