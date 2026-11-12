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


def with_import(body: str, header: str = "import mylib as m") -> list[str]:
    return [f"{header}\n{body}"]


class ContradictionTest(unittest.TestCase):
    def test_detects_contradiction(self):
        self.assertTrue(contradictory_tests([(CONTRADICTORY / "test_mylib.py").read_text(encoding="utf-8")]))

    def test_detects_mixed_form_contradiction(self):
        self.assertTrue(contradictory_tests(with_import("self.assertEqual(m.total([1, 2]), 3)\n"
                                                        "self.assertTrue(m.total([1, 2]) == 4)\n")))

    def test_detects_impossible_inequalities(self):
        for tests in (
            "self.assertNotEqual(m.total([1, 2, 3]), m.total([1, 2, 3]))\n",
            "self.assertFalse(m.total([1, 2]) == m.total([1, 2]))\n",
            "self.assertTrue(m.reverse_words('a b') != m.reverse_words('a b'))\n",
            "self.assertEqual(m.total([1, 2]), 3)\nself.assertNotEqual(m.total([1, 2]), 3)\n",
            "self.assertTrue(m.is_palindrome('abc'))\nself.assertFalse(m.is_palindrome('abc'))\n",
        ):
            with self.subTest(tests=tests):
                self.assertTrue(contradictory_tests(with_import(tests)))

    def test_detects_contradiction_with_from_import(self):
        self.assertTrue(contradictory_tests(with_import("self.assertEqual(total([1, 2]), 3)\n"
                                                        "self.assertEqual(total([1, 2]), 4)\n",
                                                        "from mylib import total")))

    def test_inequality_of_random_values_is_legit(self):
        for tests in ("self.assertNotEqual(m.random_token(), m.random_token())\n",
                      "self.assertNotEqual(m.next_id(), m.next_id())\n",
                      "self.assertNotEqual(m.total([1, 2]), 4)\n"):
            with self.subTest(tests=tests):
                self.assertFalse(contradictory_tests(with_import(tests)))

    def test_consistent_mixed_forms_are_fine(self):
        self.assertFalse(contradictory_tests(with_import("self.assertEqual(m.total([1, 2]), 3)\n"
                                                         "self.assertTrue(m.total([1, 2]) == 3)\n"
                                                         "self.assertEqual(m.average([2, 4]), 3)\n"
                                                         "self.assertEqual(m.average([2, 4]), 3.0)\n")))

    def test_variables_objects_and_state_are_not_contradictions(self):
        # Обычные выполнимые тесты, которые прежнее правило считало противоречивыми.
        cases = {
            "переменная в разных тестах": "def test_a(self):\n    result = m.total([1, 2, 3])\n"
                                          "    self.assertEqual(result, 6)\n"
                                          "def test_b(self):\n    result = m.total([])\n    self.assertEqual(result, 0)\n",
            "метод объекта с состоянием": "s = m.Stack()\ns.push(1)\ns.push(2)\n"
                                          "self.assertEqual(s.pop(), 2)\nself.assertEqual(s.pop(), 1)\n",
            "assertTrue/assertFalse переменной": "ok = m.is_palindrome('aba')\nself.assertTrue(ok)\n"
                                                 "ok = m.is_palindrome('ab')\nself.assertFalse(ok)\n",
            "вызов без аргументов": "self.assertEqual(m.tick(), 1)\nself.assertEqual(m.tick(), 2)\n",
        }
        for name, tests in cases.items():
            with self.subTest(case=name):
                self.assertFalse(contradictory_tests(with_import(tests)))

    def test_function_with_module_state_is_skipped(self):
        module = "_seen = []\n\ndef add(x):\n    _seen.append(x)\n    return len(_seen)\n"
        tests = with_import("self.assertEqual(m.add('a'), 1)\nself.assertEqual(m.add('a'), 2)\n")
        self.assertFalse(contradictory_tests(tests, module))
        self.assertTrue(contradictory_tests(tests))  # без исходника модуля — похоже на противоречие

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

    STACK = ("class Stack:\n    def __init__(self):\n        self.items = []\n\n"
             "    def pop(self):\n        return self.items.pop(0)\n\n"
             "    def size(self):\n        return len(self.items) - 1\n\n\ndef total(xs):\n    return sum(xs)\n")

    def check_stack(self, patched: str, pop_fixed: bool = True, size_fixed: bool = False) -> None:
        namespace: dict = {}
        exec(compile(patched, "mylib.py", "exec"), namespace)  # noqa: S102 — свой тестовый код
        stack = namespace["Stack"]()
        stack.items = [1, 2, 3]
        self.assertEqual(stack.pop(), 3 if pop_fixed else 1)
        self.assertEqual(stack.size(), 2 if size_fixed else 1)
        self.assertEqual(namespace["total"]([1, 2]), 3)
        self.assertEqual(patched.count("class Stack"), 1)

    def test_method_without_class(self):
        patched = apply_functions(self.STACK, "def pop(self):\n    return self.items.pop()\n")
        self.check_stack(patched)
        self.assertNotIn("\ndef pop", patched)  # не дописан отдельной функцией

    def test_partial_class(self):
        patched = apply_functions(self.STACK, "class Stack:\n    def pop(self):\n        return self.items.pop()\n")
        self.check_stack(patched)
        self.assertIn("def size", patched)  # остальные методы на месте

    def test_whole_class(self):
        fixed = self.STACK.split("\n\n\ndef total")[0].replace("pop(0)", "pop()").replace(" - 1", "")
        self.check_stack(apply_functions(self.STACK, fixed), size_fixed=True)

    def test_unusable_reply(self):
        self.assertIsNone(apply_functions("def f(x):\n    return x\n", "this is not python"))
        self.assertIsNone(apply_functions("def f(x):\n    return x\n", "x = 1\n"))


if __name__ == "__main__":
    unittest.main()
