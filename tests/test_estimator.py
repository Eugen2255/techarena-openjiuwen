"""Тесты оценщика стоимости и нормализации ответов (без LLM). Запуск: python -m unittest discover -s tests"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import estimator  # noqa: E402
from harness.answers import normalize_answer  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402

_spec = importlib.util.spec_from_file_location("fit_estimator", REPO / "scripts" / "fit_estimator.py")
fit_estimator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_estimator)


def tasks(folder: str) -> dict:
    return {t.task_id: t for t in load_tasks(REPO / folder)}


class RouteTest(unittest.TestCase):
    def test_routes_of_examples(self):
        examples = tasks("tasks_examples")
        for task_id in ("math_0000", "math_0004", "data_analysis_0002", "search_0001"):
            with self.subTest(task=task_id):
                self.assertEqual(estimator.detect_route(examples[task_id]), "fastpath")
        self.assertEqual(estimator.detect_route(examples["coding_0003"]), "coding")

    def test_contradictory_coding_is_free(self):
        task = tasks("tasks_custom")["coding_x001"]
        self.assertEqual(estimator.detect_route(task), "coding:contradiction")
        self.assertEqual(estimator.estimate(task, params={}).cost, 0)

    def test_oneshot_route(self):
        task = tasks("tasks_custom")["data_analysis_x001"]
        self.assertEqual(estimator.detect_route(task), "oneshot:data_analysis")


class PredictTest(unittest.TestCase):
    def test_defaults_without_params(self):
        cost, upper = estimator.predict("coding", {}, params={})
        self.assertEqual(cost, estimator.DEFAULT_COST["coding"])
        self.assertGreater(upper, cost)

    def test_linear_params(self):
        params = {"coding": {"intercept": 10.0, "coef": {"module_chars": 0.5}, "upper_factor": 1.5}}
        cost, upper = estimator.predict("coding", {"module_chars": 100.0}, params)
        self.assertAlmostEqual(cost, 60.0)
        self.assertAlmostEqual(upper, 90.0)

    def test_floor_for_llm_routes(self):
        params = {"coding": {"intercept": -500.0, "coef": {}, "upper_factor": 1.0}}
        self.assertEqual(estimator.predict("coding", {}, params)[0], estimator.MIN_LLM_COST)

    def test_features_of_coding_task(self):
        features = estimator.extract_features(tasks("tasks_examples")["coding_0003"])
        self.assertEqual(features["n_functions"], 6)
        self.assertEqual(features["n_asserts"], 6)
        self.assertGreater(features["module_chars"], 300)


class FitTest(unittest.TestCase):
    def test_recovers_linear_relation(self):
        rows = [[float(x)] for x in range(1, 21)]
        y = [3.0 + 2.0 * r[0] for r in rows]
        intercept, coef = fit_estimator.fit_linear(rows, y, ridge=1e-6)
        self.assertAlmostEqual(intercept, 3.0, places=3)
        self.assertAlmostEqual(coef[0], 2.0, places=3)

    def test_mean_model(self):
        intercept, coef = fit_estimator.fit_linear([[], [], []], [1.0, 2.0, 3.0])
        self.assertEqual((intercept, coef), (2.0, []))


class NormalizeAnswerTest(unittest.TestCase):
    def test_exact_int(self):
        self.assertEqual(normalize_answer("6668.0", "exact_int"), "6668")
        self.assertEqual(normalize_answer("1,234", "exact_int"), "1234")
        self.assertEqual(normalize_answer(" 42 ", "exact_int"), "42")
        self.assertEqual(normalize_answer("12.5", "exact_int"), "12.5")

    def test_exact_str_untouched(self):
        self.assertEqual(normalize_answer("1,234", "exact_str"), "1,234")


if __name__ == "__main__":
    unittest.main()
