"""Тесты решателя math без LLM. Запуск из корня репозитория: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.fastpath import try_solve  # noqa: E402
from harness.tasks import Task, load_tasks  # noqa: E402

TAIL = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, "
        "write ONLY the final answer to the file answer.txt in the current directory "
        "(for example with bash: echo 42 > answer.txt).")


def solve(text: str):
    task = Task(task_id="t", task_dir=REPO, task_type="math", instruction=text + TAIL,
                reward=1, check="exact_int", raw={})
    return try_solve(task, REPO)


class ExamplesTest(unittest.TestCase):
    def test_organizer_examples(self):
        tasks = [t for t in load_tasks(REPO / "tasks_examples") if t.task_type == "math"]
        self.assertTrue(tasks)
        for task in tasks:
            with self.subTest(task=task.task_id):
                result = try_solve(task, REPO)
                self.assertIsNotNone(result)
                self.assertEqual(result.answer, task.raw["expected"])

    def test_custom_infeasible(self):
        for task in load_tasks(REPO / "tasks_custom"):
            if task.task_type == "math":
                with self.subTest(task=task.task_id):
                    result = try_solve(task, REPO)
                    self.assertIsNotNone(result)
                    self.assertTrue(result.infeasible)
                    self.assertIsNone(result.answer)


class VariantsTest(unittest.TestCase):
    SOLVED = [
        ("A warehouse starts with 100 boxes. Then the stock is split equally into 3 warehouses and it keeps "
         "one share (rounded up). How many boxes does it have at the end?", "34"),
        ("A warehouse starts with 10 boxes. Then the stock is tripled by merging with two sister warehouses. "
         "It receives 3 boxes. How many boxes does it have at the end?", "33"),
        ("A warehouse starts with 500 boxes. It ships 20 boxes per day for 1 day. Next, it sells 30 boxes. "
         "How many boxes does it have at the end?", "450"),
        ("A depot starts with 7 crates. It receives 1 delivery of 5 crates each. "
         "How many crates does it have at the end?", "12"),
    ]
    INFEASIBLE = {
        "unknown start": "A warehouse starts with an unknown number of boxes. It ships 36 boxes per day for 3 days. "
                         "How many boxes does it have at the end?",
        "vague shipment": "A warehouse starts with 500 boxes. It ships some boxes per day for 3 days. "
                          "How many boxes does it have at the end?",
        "vague split": "A warehouse starts with 500 boxes. Then the stock is split equally into several warehouses "
                       "and it keeps one share (rounded down). How many boxes does it have at the end?",
    }
    REFUSED = {
        "vague but unknown operation": "A warehouse starts with 100 boxes. Then some of the stock is lost. "
                                       "How many boxes does it have at the end?",
        "negative stock": "A warehouse starts with 10 boxes. It ships 5 boxes per day for 3 days. "
                          "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?",
        "unknown operation": "A warehouse starts with 100 boxes. Then 10% of the stock is lost. "
                             "How many boxes does it have at the end?",
        "split into zero": "A warehouse starts with 100 boxes. Then the stock is split equally into 0 warehouses "
                           "and it keeps one share (rounded down). How many boxes does it have at the end?",
        "no rounding rule": "A warehouse starts with 100 boxes. Then the stock is split equally into 3 warehouses "
                            "and it keeps one share. How many boxes does it have at the end?",
        "other unit": "A warehouse starts with 100 boxes. It receives 2 deliveries of 10 crates each. "
                      "How many boxes does it have at the end?",
        "other question": "A warehouse starts with 100 boxes. It ships 10 boxes per day for 2 days. "
                          "How many boxes did it ship?",
    }

    def test_solved(self):
        for text, expected in self.SOLVED:
            with self.subTest(text=text[:60]):
                result = solve(text)
                self.assertIsNotNone(result)
                self.assertEqual(result.answer, expected)

    def test_infeasible(self):
        for name, text in self.INFEASIBLE.items():
            with self.subTest(case=name):
                result = solve(text)
                self.assertIsNotNone(result)
                self.assertTrue(result.infeasible)
                self.assertIsNone(result.answer)

    def test_refused(self):
        for name, text in self.REFUSED.items():
            with self.subTest(case=name):
                self.assertIsNone(solve(text))


if __name__ == "__main__":
    unittest.main()
