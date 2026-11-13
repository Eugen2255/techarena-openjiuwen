"""Устойчивость к непривычному виду задач и ответов (без LLM). Запуск: python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.answers import is_infeasible, normalize_answer  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402


class NormalizeAnswerTest(unittest.TestCase):
    def test_wrappers_and_trailing_period(self):
        # Организаторы сравнивают после strip и нижнего регистра: обёртки и точку снимаем сами.
        for raw, expected in {"Osaka.": "Osaka", "**Osaka**": "Osaka", "`42`": "42", '"Osaka"': "Osaka",
                              "'Lyon'": "Lyon", "«Porto»": "Porto", "**`Seoul`**": "Seoul", " 1983. ": "1983",
                              "Acme Inc.": "Acme Inc.", "Osaka, Japan": "Osaka, Japan", "data engineer": "data engineer",
                              "Pavel Muller": "Pavel Muller"}.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw, "exact_str"), expected)

    def test_exact_int_forms(self):
        for raw, expected in {"6668.0": "6668", "1,234": "1234", "$1,200": "1200", "42 boxes": "42",
                              "The answer is 42": "42", "-5": "-5", "**389**": "389", "1735.": "1735",
                              "3.5": "3.5", "3 days and 4 hours": "3 days and 4 hours"}.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw, "exact_int"), expected)

    def test_infeasible_survives(self):
        for raw in ("INFEASIBLE", "`INFEASIBLE`", "**INFEASIBLE**"):
            with self.subTest(raw=raw):
                self.assertTrue(is_infeasible(normalize_answer(raw, "exact_str")))


class LoaderTest(unittest.TestCase):
    def write(self, root: Path, name: str, data) -> None:
        (root / name).mkdir()
        (root / name / "task.json").write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def test_odd_task_json_does_not_break_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write(root, "math_0001", {"task_type": "Math", "instruction": "x", "reward": "3 points"})
            self.write(root, "data_analysis_0002", {"task_type": "Data-Analysis", "instruction": "x", "reward": 2.5,
                                                    "check": "exact_int"})
            self.write(root, "search_0003", {"instruction": "x"})  # нет типа и награды
            self.write(root, "coding_0004", ["not", "an", "object"])
            self.write(root, "broken_0005", "{not json")
            self.write(root, "math_0006", {"task_type": "math", "reward": 0, "check": 5})
            tasks = {t.task_id: t for t in load_tasks(root)}
        self.assertEqual(sorted(tasks), ["data_analysis_0002", "math_0001", "math_0006", "search_0003"])
        self.assertEqual((tasks["math_0001"].task_type, tasks["math_0001"].reward), ("math", 3.0))
        self.assertEqual((tasks["data_analysis_0002"].task_type, tasks["data_analysis_0002"].reward),
                         ("data_analysis", 2.5))
        self.assertEqual((tasks["search_0003"].task_type, tasks["search_0003"].reward), ("search", 1.0))
        self.assertEqual((tasks["math_0006"].reward, tasks["math_0006"].check), (0.0, None))


if __name__ == "__main__":
    unittest.main()
