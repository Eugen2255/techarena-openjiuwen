"""Тесты помощников coding (без LLM). Запуск из корня репозитория: python -m unittest discover -s tests"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.coding import apply_functions, compact_test_output, contradictory_tests  # noqa: E402

EXAMPLE = REPO / "tasks_examples" / "coding_0003" / "workspace"
CONTRADICTORY = REPO / "tasks_custom" / "coding_x001" / "workspace"

FIX = '''```python
def reverse_words(s):
    return ' '.join(reversed(s.split()))


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s
```'''


def run_tests(workdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "test_mylib.py"], cwd=workdir, capture_output=True, text=True, timeout=60)


class CompactOutputTest(unittest.TestCase):
    def test_keeps_failures_and_drops_noise(self):
        with tempfile.TemporaryDirectory() as tmp:  # копия, чтобы не мусорить в tasks_examples
            shutil.copytree(EXAMPLE, tmp, dirs_exist_ok=True)
            output = run_tests(Path(tmp)).stderr
        compact = compact_test_output(output)
        self.assertIn("FAIL: test_reverse_words", compact)
        self.assertIn("self.assertEqual(m.reverse_words('c a b'), 'b a c')", compact)
        self.assertIn("AssertionError: 'a b c' != 'b a c'", compact)
        self.assertNotIn("Traceback", compact)
        self.assertNotIn('File "', compact)
        self.assertLess(len(compact), len(output) / 2)


class ContradictionTest(unittest.TestCase):
    def test_detects_contradiction(self):
        self.assertTrue(contradictory_tests([(CONTRADICTORY / "test_mylib.py").read_text(encoding="utf-8")]))

    def test_detects_mixed_form_contradiction(self):
        self.assertTrue(contradictory_tests(["self.assertEqual(m.total([1, 2]), 3)\n"
                                             "self.assertTrue(m.total([1, 2]) == 4)\n"]))

    def test_consistent_mixed_forms_are_fine(self):
        self.assertFalse(contradictory_tests(["self.assertEqual(m.total([1, 2]), 3)\n"
                                              "self.assertTrue(m.total([1, 2]) == 3)\n"]))

    def test_normal_tests_are_not_contradictory(self):
        for task in ("coding_0003", "coding_0007"):
            with self.subTest(task=task):
                text = (REPO / "tasks_examples" / task / "workspace" / "test_mylib.py").read_text(encoding="utf-8")
                self.assertFalse(contradictory_tests([text]))


class ApplyFunctionsTest(unittest.TestCase):
    def test_fix_makes_tests_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            shutil.copytree(EXAMPLE, workdir, dirs_exist_ok=True)
            source = (workdir / "mylib.py").read_text(encoding="utf-8")
            patched = apply_functions(source, FIX.split("```python\n")[1].split("```")[0])
            self.assertIsNotNone(patched)
            (workdir / "mylib.py").write_text(patched, encoding="utf-8")
            self.assertEqual(run_tests(workdir).returncode, 0)
            self.assertIn("def count_even", patched)  # остальные функции на месте

    def test_new_function_and_import(self):
        patched = apply_functions("def f(x):\n    return x\n",
                                  "import math\n\ndef f(x):\n    return helper(x)\n\ndef helper(x):\n    return math.floor(x)\n")
        self.assertTrue(patched.startswith("import math\n"))
        self.assertIn("return helper(x)", patched)
        self.assertIn("def helper(x):", patched)
        self.assertEqual(patched.count("def f("), 1)

    def test_unusable_reply(self):
        self.assertIsNone(apply_functions("def f(x):\n    return x\n", "this is not python"))
        self.assertIsNone(apply_functions("def f(x):\n    return x\n", "x = 1\n"))


if __name__ == "__main__":
    unittest.main()
