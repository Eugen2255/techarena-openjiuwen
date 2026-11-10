"""Тесты решателей data_analysis без LLM. Запуск из корня репозитория: python -m unittest discover -s tests"""
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.fastpath import try_solve  # noqa: E402
from harness.tasks import Task, load_tasks  # noqa: E402

TAIL = (" Use bash with python3 to compute it. Work only inside the current directory. When you are done, write ONLY "
        "the final answer to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CSV = ("order_id,region,product,month,units,price\n"
       "1,north,widget,1,10,5\n2,south,widget,2,7,5\n3,east,gadget,2,4,9\n4,west,widget,5,3,5\n")


def solve(instruction: str, files: dict[str, str]):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, text in files.items():
            (workdir / name).write_text(text, encoding="utf-8")
        task = Task(task_id="t", task_dir=workdir, task_type="data_analysis", instruction=instruction + TAIL,
                    reward=1, check="exact_int", raw={})
        return try_solve(task, workdir)


HEAD = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "


class DataAnalysisTest(unittest.TestCase):
    def test_organizer_examples(self):
        tasks = [t for t in load_tasks(REPO / "tasks_examples") if t.task_type == "data_analysis"]
        self.assertTrue(tasks)
        for task in tasks:
            with self.subTest(task=task.task_id):
                result = try_solve(task, task.workspace)
                self.assertIsNotNone(result)
                self.assertEqual(result.answer, task.raw["expected"])

    def test_units_in_month_range(self):
        result = solve(HEAD + "How many units of product 'widget' were sold in months 1 through 2 inclusive?",
                       {"sales.csv": CSV})
        self.assertEqual(result.answer, "17")

    def test_zero_units_in_range_is_an_answer(self):
        result = solve(HEAD + "How many units of product 'gadget' were sold in months 6 through 8 inclusive?",
                       {"sales.csv": CSV})
        self.assertEqual(result.answer, "0")

    def test_missing_file_is_infeasible(self):
        result = solve("The file returns.csv has columns order_id,product,month,units (3 rows). How many units of "
                       "product 'widget' were returned in months 1 through 2 inclusive?", {"sales.csv": CSV})
        self.assertIsNotNone(result)
        self.assertTrue(result.infeasible)

    def test_refused(self):
        cases = {
            "other question": HEAD + "What was the total revenue (units times price) from product 'widget'?",
            "unknown product": HEAD + "How many units of product 'gizmo' were sold in months 1 through 2 inclusive?",
            "bad month range": HEAD + "How many units of product 'widget' were sold in months 5 through 2 inclusive?",
        }
        for name, instruction in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(solve(instruction, {"sales.csv": CSV}))


if __name__ == "__main__":
    unittest.main()
