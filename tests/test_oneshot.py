"""Тесты помощников решения одним вызовом (без LLM). Запуск: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import oneshot  # noqa: E402
from harness.answers import is_infeasible, question_text, says_infeasible  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402

EXAMPLES = {t.task_id: t for t in load_tasks(REPO / "tasks_examples")}
CUSTOM = {t.task_id: t for t in load_tasks(REPO / "tasks_custom")}


class QuestionTextTest(unittest.TestCase):
    def test_strips_service_tail(self):
        for task in EXAMPLES.values():
            with self.subTest(task=task.task_id):
                text = question_text(task.instruction)
                self.assertNotIn("answer.txt", text)
                self.assertNotIn("Work only inside", text)
                self.assertTrue(text)

    def test_keeps_question(self):
        self.assertTrue(question_text(EXAMPLES["search_0001"].instruction)
                        .endswith("In which year was Irina Smith born?"))
        self.assertTrue(question_text(EXAMPLES["math_0004"].instruction)
                        .endswith("How many boxes does it have at the end?"))


class KnowledgeBaseTest(unittest.TestCase):
    def test_one_line_per_entry(self):
        task = EXAMPLES["search_0001"]
        files = oneshot.kb_files(task)
        kb = oneshot.compact_kb(files)
        self.assertEqual(len(kb.splitlines()), len(files))
        self.assertIn("Irina Smith | Born: 1983; Employer: Dune Robotics; Mentor: Alice Smith", kb)
        self.assertIn("Granite AI | Headquarters: Osaka; Founded: 1992", kb)


class SupportTest(unittest.TestCase):
    def test_supported_types(self):
        for task_id in ("math_0000", "data_analysis_0002", "search_0005"):
            with self.subTest(task=task_id):
                self.assertTrue(oneshot.supports_oneshot(EXAMPLES[task_id]))
        self.assertTrue(oneshot.supports_oneshot(CUSTOM["data_analysis_x001"]))
        self.assertFalse(oneshot.supports_oneshot(EXAMPLES["coding_0003"]))  # у coding своя схема


class MissingQuantityTest(unittest.TestCase):
    COLUMNS = ["order_id", "region", "product", "month", "units", "price"]

    def test_quantity_without_column(self):
        question = "What was the total discount given on product 'widget' in months 7 through 8 inclusive?"
        self.assertEqual(oneshot.missing_quantity(question, self.COLUMNS), "discount")
        self.assertEqual(oneshot.missing_quantity("What is the average rating of 'gizmo'?", self.COLUMNS), "rating")

    def test_feasible_questions_pass(self):
        for question in (
            "What was the total revenue (units times price) from product 'widget' in months 1 through 3 inclusive?",
            "How many units were sold in region 'north' in months 2 through 4 inclusive?",
            "How many orders of product 'gizmo' were placed in the 'east' region?",
            "What is the total number of units sold in month 5?",
            "What is the average price of product 'widget'?",
            "What is the total units of product 'widget'?",
            "What is the total sales of product 'widget'?",
        ):
            with self.subTest(question=question):
                self.assertIsNone(oneshot.missing_quantity(question, self.COLUMNS))


class ReplyParsingTest(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(oneshot.extract_code("text\n```python\nprint(1)\n```\nmore"), "print(1)")
        self.assertIsNone(oneshot.extract_code("print(1)"))
        self.assertIsNone(oneshot.extract_code("```python\n\n```"))

    def test_key_error_column(self):
        output = "Traceback (most recent call last):\n  ...\nKeyError: 'discount'\n"
        self.assertEqual(oneshot._KEY_ERROR_RE.search(output)["key"], "discount")

    def test_infeasible_markers(self):
        for reply in ("INFEASIBLE", "infeasible.", "`INFEASIBLE`", " Infeasible "):
            with self.subTest(reply=reply):
                self.assertTrue(is_infeasible(reply))
        self.assertFalse(is_infeasible("42"))

    def test_says_infeasible_with_explanation(self):
        self.assertTrue(says_infeasible("The file has no discount column, so it cannot be computed.\n\nINFEASIBLE"))
        self.assertTrue(says_infeasible("INFEASIBLE"))
        # Код в ответе важнее упоминания слова (например, в комментарии): его запустим.
        self.assertFalse(says_infeasible("```python\n# not INFEASIBLE\nprint(1)\n```"))
        self.assertFalse(says_infeasible("The answer is 42"))


if __name__ == "__main__":
    unittest.main()
