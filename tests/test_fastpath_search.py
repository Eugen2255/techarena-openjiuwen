"""Тесты решателя search без LLM. Запуск из корня репозитория: python -m unittest discover -s tests"""
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.fastpath import try_solve  # noqa: E402
from harness.tasks import Task, load_tasks  # noqa: E402

HEAD = "The directory kb/ contains a small knowledge base of markdown files. "
TAIL = (" Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. When you "
        "are done, write ONLY the final answer to the file answer.txt in the current directory "
        "(for example with bash: echo 42 > answer.txt).")

BASE_KB = {
    "person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\nMentor: Alice Smith\n",
    "person_alice_smith.md": "# Alice Smith\n\nBorn: 1961\nEmployer: Granite AI\nMentor: Olga Kowalski\n",
    "person_olga_kowalski.md": "# Olga Kowalski\n\nBorn: 1950\nEmployer: Granite AI\nMentor: Alice Smith\n",
    "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Oslo\nFounded: 1999\n",
    "company_granite_ai.md": "# Granite AI\n\nHeadquarters: Lisbon\nFounded: 1982\n",
}


def solve(question: str, kb: dict[str, str] | None = None):
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "kb").mkdir()
        for name, text in (BASE_KB if kb is None else kb).items():
            (workdir / "kb" / name).write_text(text, encoding="utf-8")
        task = Task(task_id="t", task_dir=workdir, task_type="search", instruction=HEAD + question + TAIL,
                    reward=1, check="exact_str", raw={})
        return try_solve(task, workdir)


class ExamplesTest(unittest.TestCase):
    def test_organizer_examples(self):
        tasks = [t for t in load_tasks(REPO / "tasks_examples") if t.task_type == "search"]
        self.assertTrue(tasks)
        for task in tasks:
            with self.subTest(task=task.task_id):
                result = try_solve(task, task.workspace)
                self.assertIsNotNone(result)
                self.assertEqual(result.answer, task.raw["expected"])

    def test_custom_infeasible(self):
        for task in load_tasks(REPO / "tasks_custom"):
            if task.task_type == "search":
                with self.subTest(task=task.task_id):
                    result = try_solve(task, task.workspace)
                    self.assertIsNotNone(result)
                    self.assertTrue(result.infeasible)
                    self.assertIsNone(result.answer)


class VariantsTest(unittest.TestCase):
    SOLVED = [
        ("In which year was Irina Smith born?", "1983"),
        ("When was Alice Smith born?", "1961"),
        ("In which city is the headquarters of Irina Smith's employer?", "Oslo"),
        ("In which year was Irina Smith's employer founded?", "1999"),
        ("Who is Irina Smith's mentor?", "Alice Smith"),
        ("Which company does Irina Smith work for?", "Dune Robotics"),
        ("In which city is the headquarters of Irina Smith's mentor's employer?", "Lisbon"),
        ("In which year was the mentor of Irina Smith born?", "1961"),
        ("Where is Granite AI headquartered?", "Lisbon"),
    ]

    def test_solved(self):
        for question, expected in self.SOLVED:
            with self.subTest(question=question):
                result = solve(question)
                self.assertIsNotNone(result)
                self.assertFalse(result.infeasible)
                self.assertEqual(result.answer, expected)

    def test_infeasible(self):
        dangling = dict(BASE_KB)
        del dangling["company_dune_robotics.md"]
        no_mentor = dict(BASE_KB, **{"person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\n"})
        twins = dict(BASE_KB, **{"person_irina_smith_2.md": "# Irina Smith\n\nBorn: 1990\nEmployer: Granite AI\n"
                                                            "Mentor: Alice Smith\n"})
        two_irinas = dict(BASE_KB, **{"person_irina_haddad.md": "# Irina Haddad\n\nBorn: 1975\nEmployer: Granite AI\n"
                                                                "Mentor: Alice Smith\n"})
        cases = {
            "person absent": ("In which year was Pavel Haddad born?", None),
            "employer has no file": ("In which city is the headquarters of Irina Smith's employer?", dangling),
            "missing field": ("Who is Irina Smith's mentor?", no_mentor),
            "namesakes disagree": ("In which year was Irina Smith born?", twins),
            "first name of several people": ("In which year was Irina born?", two_irinas),
        }
        for name, (question, kb) in cases.items():
            with self.subTest(case=name):
                result = solve(question, kb)
                self.assertIsNotNone(result)
                self.assertTrue(result.infeasible)
                self.assertIsNone(result.answer)

    def test_namesakes_that_agree_are_answered(self):
        twins = dict(BASE_KB, **{"person_irina_smith_2.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Granite AI\n"
                                                            "Mentor: Alice Smith\n"})
        self.assertEqual(solve("In which year was Irina Smith born?", twins).answer, "1983")

    def test_refused(self):
        unknown_field = dict(BASE_KB, **{"person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmail: i@x.org\n"})
        cases = {
            "unknown question": ("How old is Irina Smith?", None),
            "who is without relation": ("Who is Irina Smith?", None),
            "first name only": ("In which year was Irina born?", None),
            "lowercase name": ("In which year was irina smith born?", None),
            "unknown relation": ("Who is Irina Smith's manager?", None),
            "unexpected kb format + absent person": ("In which year was Pavel Haddad born?", unknown_field),
            "empty kb": ("In which year was Irina Smith born?", {}),
        }
        for name, (question, kb) in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(solve(question, kb))


if __name__ == "__main__":
    unittest.main()
