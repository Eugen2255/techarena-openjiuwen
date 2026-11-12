#!/usr/bin/env python3
"""Генератор тренировочных задач в формате организаторов: 4 типа, включая невыполнимые.

    python scripts/generate_tasks.py --count 100 --seed 1 --out tasks_generated
    python scripts/generate_tasks.py --count 300 --infeasible-share 0.25 --paraphrase-share 0.4 --force

Каждая задача — <out>/<type>_<NNNN>/task.json (+ workspace/). Поля как у организаторов, плюс:
  "expected": ответ (null для невыполнимых), "feasible": false для невыполнимых,
  "generator": {"variant": ...} — шаблон из примеров, перефразировка или вид невыполнимости.
Харнесс эти поля не читает, их использует scripts/evaluate.py и scripts/check_fastpath.py.
Скрытые тесты coding лежат в <task>/hidden/ (вне workspace, как у организаторов); их запускает evaluate.py.

Перефразировки специально включают формулировки, которые решатели быстрого пути не разбирают:
так проверяется, что задача уходит агенту, а не получает неверный ответ.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

WORK_TAIL = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
             "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING_INSTRUCTION = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
                      "(do not edit the tests) until all tests pass. Work only inside the current directory. When you "
                      "are done, write ONLY the word DONE to the file answer.txt in the current directory (for example "
                      "with bash: echo 42 > answer.txt).")


@dataclass
class Generated:
    task_type: str
    instruction: str
    expected: str | None
    check: str
    variant: str
    files: dict[str, str] = field(default_factory=dict)   # путь внутри workspace -> текст
    hidden: dict[str, str] = field(default_factory=dict)  # скрытые тесты (вне workspace)

    @property
    def feasible(self) -> bool:
        return self.expected is not None


# ---------------------------------------------------------------- math

def gen_math(rng: random.Random, infeasible: str | None, paraphrase: bool) -> Generated:
    tail = " Compute it with the bash tool (python3 -c). " + WORK_TAIL
    template_ops = ["split", "double", "ship", "receive"]
    extra_ops = ["sell", "more", "triple", "halve", "damage"]  # halve и damage решатель не разбирает
    while True:
        start = rng.randint(100, 2000)
        stock, sentences, used_extra = start, [], False
        for _ in range(rng.randint(1, 5)):
            op = rng.choice(template_ops + (extra_ops if paraphrase else []))
            used_extra |= op in extra_ops
            if op == "split":
                k = rng.randint(2, 4)
                sentences.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).")
                stock //= k
            elif op == "double":
                sentences.append("Then the stock is doubled by merging with a sister warehouse.")
                stock *= 2
            elif op == "ship":
                a, d = rng.randint(5, 60), rng.randint(1, 5)
                sentences.append(f"It ships {a} boxes per day for {d} days.")
                stock -= a * d
            elif op == "receive":
                c, s = rng.randint(1, 4), rng.randint(10, 120)
                sentences.append(f"It receives {c} deliveries of {s} boxes each.")
                stock += c * s
            elif op == "sell":
                n = rng.randint(5, 80)
                sentences.append(f"It sells {n} boxes.")
                stock -= n
            elif op == "more":
                n = rng.randint(5, 80)
                sentences.append(f"It receives {n} more boxes.")
                stock += n
            elif op == "triple":
                sentences.append("Then the stock is tripled by merging with two sister warehouses.")
                stock *= 3
            elif op == "halve":
                sentences.append("Then the stock is halved (rounded down).")
                stock //= 2
            elif op == "damage":
                n = rng.randint(5, 40)
                sentences.append(f"It loses {n} boxes to water damage.")
                stock -= n
            if stock < 0:
                break
        if stock >= 0:
            break

    head = f"A warehouse starts with {start} boxes."
    expected, variant = str(stock), "paraphrase" if used_extra else "template"
    if infeasible == "unknown_start":
        head = "A warehouse starts with an unknown number of boxes."
    elif infeasible == "vague_quantity":
        sentences.append(rng.choice([f"It ships some boxes per day for {rng.randint(2, 5)} days.",
                                     f"It receives several deliveries of {rng.randint(10, 90)} boxes each."]))
    elif infeasible == "zero_split":
        sentences.append("Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).")
    elif infeasible == "missing_duration":  # сколько дней — не сказано
        sentences.insert(rng.randint(0, len(sentences)),
                         rng.choice([f"It ships {rng.randint(5, 60)} boxes per day.",
                                     f"It ships {rng.randint(5, 60)} boxes every day."]))
    elif infeasible == "missing_parts":  # на сколько складов делят — не сказано
        sentences.insert(rng.randint(0, len(sentences)),
                         "Then the stock is split equally among its sister warehouses and it keeps one share "
                         "(rounded down).")
    if infeasible:
        expected, variant = None, f"infeasible:{infeasible}"
    text = " ".join([head, *sentences, "How many boxes does it have at the end?"]) + tail
    return Generated("math", text, expected, "exact_int", variant)


# ---------------------------------------------------------------- data_analysis

PRODUCTS = ["widget", "gadget", "sprocket", "gizmo", "doohickey"]
REGIONS = ["north", "south", "east", "west"]


def gen_data_analysis(rng: random.Random, infeasible: str | None, paraphrase: bool) -> Generated:
    n_rows = rng.choice([20, 40, 80, 150, 300])
    products = rng.sample(PRODUCTS, rng.randint(3, 5))
    rows = [{"order_id": i + 1, "region": rng.choice(REGIONS), "product": rng.choice(products),
             "month": rng.randint(1, 12), "units": rng.randint(1, 40), "price": rng.randint(5, 100)}
            for i in range(n_rows)]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    files = {"sales.csv": out.getvalue()}

    product = rng.choice(sorted({r["product"] for r in rows}))
    region = rng.choice(REGIONS)
    first = rng.randint(1, 11)
    last = rng.randint(first, min(12, first + 3))
    in_range = [r for r in rows if first <= r["month"] <= last]
    head = f"The file sales.csv has columns order_id,region,product,month,units,price ({n_rows} rows). "

    kind = "template"
    question = f"How many units of product '{product}' were sold in months {first} through {last} inclusive?"
    expected: str | None = str(sum(r["units"] for r in in_range if r["product"] == product))
    if paraphrase and not infeasible:
        kind = rng.choice(["revenue", "region_units", "order_count"])
        if kind == "revenue":
            question = (f"What was the total revenue (units times price) from product '{product}' in months "
                        f"{first} through {last} inclusive?")
            expected = str(sum(r["units"] * r["price"] for r in in_range if r["product"] == product))
        elif kind == "region_units":
            question = f"How many units were sold in region '{region}' in months {first} through {last} inclusive?"
            expected = str(sum(r["units"] for r in in_range if r["region"] == region))
        else:
            question = f"How many orders of product '{product}' were placed in the '{region}' region?"
            expected = str(sum(1 for r in rows if r["product"] == product and r["region"] == region))
        kind = f"paraphrase:{kind}"

    if infeasible == "missing_column":
        question = (f"What was the total discount given on product '{product}' in months {first} through {last} "
                    "inclusive?")
    elif infeasible == "missing_file":
        head = f"The file returns.csv has columns order_id,product,month,units ({n_rows // 4} rows). "
        question = f"How many units of product '{product}' were returned in months {first} through {last} inclusive?"
    elif infeasible == "returns_not_tracked":  # почти шаблон, но в файле только продажи
        question = f"How many units of product '{product}' were returned in months {first} through {last} inclusive?"
    elif infeasible == "missing_column_phrasing":  # колонки нет, но и слов total/average нет
        question = rng.choice([
            f"How much profit did product '{product}' make in months {first} through {last} inclusive?",
            f"Which customer bought the most units of product '{product}'?",
            f"What was the shipping cost of order {rng.randint(1, n_rows)}?",
        ])
    if infeasible:
        expected, kind = None, f"infeasible:{infeasible}"
    text = head + question + " Use bash with python3 to compute it. " + WORK_TAIL
    return Generated("data_analysis", text, expected, "exact_int", kind, files)


# ---------------------------------------------------------------- search

FIRST_NAMES = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya",
               "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST_NAMES = ["Garcia", "Haddad", "Ivanova", "Kowalski", "Muller", "Novak", "Petrov", "Sato", "Smith", "Wang"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Oslo", "Lisbon", "Osaka", "Berlin", "Toronto", "Seoul", "Austin", "Lyon", "Porto", "Kyoto", "Helsinki",
          "Madrid", "Vienna", "Denver"]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def gen_search(rng: random.Random, infeasible: str | None, paraphrase: bool) -> Generated:
    companies = {c: {"Headquarters": rng.choice(CITIES), "Founded": str(rng.randint(1950, 2015))} for c in COMPANIES}
    names = rng.sample([f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES], rng.randint(24, 30))
    people = {n: {"Born": str(rng.randint(1950, 2000)), "Employer": rng.choice(COMPANIES)} for n in names}
    for n in names:
        people[n]["Mentor"] = rng.choice([m for m in names if m != n])
    person = rng.choice(names)
    p, mentor = people[person], people[people[person]["Mentor"]]

    templates = [
        ("template:born", f"In which year was {person} born?", p["Born"]),
        ("template:hq_of_employer", f"In which city is the headquarters of {person}'s employer?",
         companies[p["Employer"]]["Headquarters"]),
    ]
    paraphrases = [
        ("paraphrase:mentor", f"Who is {person}'s mentor?", p["Mentor"]),
        ("paraphrase:founded", f"In which year was {person}'s employer founded?", companies[p["Employer"]]["Founded"]),
        ("paraphrase:mentor_employer_hq", f"In which city is the headquarters of {person}'s mentor's employer?",
         companies[mentor["Employer"]]["Headquarters"]),
        ("paraphrase:mentor_born", f"In which year was {person}'s mentor born?", mentor["Born"]),
        # Эти две формулировки решатель не разбирает — должны уйти агенту.
        ("paraphrase:based_in", f"What city is {person}'s employer based in?", companies[p["Employer"]]["Headquarters"]),
        ("paraphrase:name_company", f"Name the company that employs {person}.", p["Employer"]),
    ]
    variant, question, expected = rng.choice(paraphrases if paraphrase else templates)

    extra_files: dict[str, str] = {}
    if infeasible == "absent_person":
        absent = rng.choice([f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES if f"{f} {l}" not in people])
        question = f"In which year was {absent} born?"
    elif infeasible == "dangling_employer":
        question = f"In which city is the headquarters of {person}'s employer?"
        del companies[p["Employer"]]
    elif infeasible == "missing_mentor":
        question = f"Who is {person}'s mentor?"
        del p["Mentor"]
    elif infeasible == "namesakes":
        question = f"In which year was {person} born?"
        twin_born = str(int(p["Born"]) + rng.randint(1, 15))
        extra_files[f"kb/person_{_slug(person)}_2.md"] = (
            f"# {person}\n\nBorn: {twin_born}\nEmployer: {rng.choice(COMPANIES)}\nMentor: {rng.choice(names)}\n")
    elif infeasible == "unknown_field":  # такого поля в базе нет ни у кого
        question = rng.choice([f"What is {person}'s email address?", f"What is the phone number of {person}?",
                               f"Which university did {person} graduate from?"])
    elif infeasible == "absent_company":
        company = rng.choice(["Nova Dynamics", "Zephyr Labs", "Orion Freight", "Quasar Textiles"])
        question = rng.choice([f"In which city is the headquarters of {company}?",
                               f"In which year was {company} founded?"])
    elif infeasible == "ambiguous_first_name":  # имя без фамилии, а таких людей несколько
        counts: dict[str, int] = {}
        for n in names:
            counts[n.split()[0]] = counts.get(n.split()[0], 0) + 1
        shared = sorted(f for f, c in counts.items() if c >= 2)
        if not shared:  # редкий случай: добавим однофамильца по имени
            first_name = person.split()[0]
            twin = next(f"{first_name} {l}" for l in LAST_NAMES if f"{first_name} {l}" not in people)
            people[twin] = {"Born": str(rng.randint(1950, 2000)), "Employer": rng.choice(COMPANIES),
                            "Mentor": person}
            shared = [first_name]
        question = f"In which year was {rng.choice(shared)} born?"
    if infeasible:
        expected, variant = None, f"infeasible:{infeasible}"

    files = {f"kb/company_{_slug(c)}.md": f"# {c}\n\n" + "".join(f"{k}: {v}\n" for k, v in fields.items())
             for c, fields in companies.items()}
    files.update({f"kb/person_{_slug(n)}.md": f"# {n}\n\n" + "".join(f"{k}: {v}\n" for k, v in fields.items())
                  for n, fields in people.items()})
    files.update(extra_files)
    text = ("The directory kb/ contains a small knowledge base of markdown files. " + question
            + " Use the file tools or bash (ls, grep, cat) to find it. " + WORK_TAIL)
    return Generated("search", text, expected, "exact_str", variant, files)


# ---------------------------------------------------------------- coding

# Функция: правильная реализация, варианты с багом, видимые и скрытые проверки.
FUNCTIONS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                       "def reverse_words(s):\n    return ' '.join(s.split())\n"],
                      ["self.assertEqual(m.reverse_words('c a b'), 'b a c')"],
                      ["self.assertEqual(m.reverse_words('one two three'), 'three two one')",
                       "self.assertEqual(m.reverse_words('x'), 'x')"]),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
               "def total(xs):\n    s = 0\n    for x in xs:\n        s = x\n    return s\n"],
              ["self.assertEqual(m.total([1, 2, 3]), 6)"],
              ["self.assertEqual(m.total([]), 0)", "self.assertEqual(m.total([5, -2]), 3)"]),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"],
                ["self.assertEqual(m.maximum([3, 9, 2]), 9)"],
                ["self.assertEqual(m.maximum([-5, -2, -9]), -2)", "self.assertEqual(m.maximum([7]), 7)"]),
    "minimum": ("def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                ["def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n"],
                ["self.assertEqual(m.minimum([3, 9, 2]), 2)"],
                ["self.assertEqual(m.minimum([4, 4, 1, 7]), 1)", "self.assertEqual(m.minimum([-1, -8]), -8)"]),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                ["def average(xs):\n    return sum(xs) / (len(xs) + 1)\n"],
                ["self.assertEqual(m.average([2, 4, 6]), 4)"],
                ["self.assertEqual(m.average([1, 2]), 1.5)", "self.assertEqual(m.average([5]), 5)"]),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                ["def first_n(xs, n):\n    return xs[:n - 1]\n", "def first_n(xs, n):\n    return xs[1:n + 1]\n"],
                ["self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"],
                ["self.assertEqual(m.first_n([1, 2], 5), [1, 2])", "self.assertEqual(m.first_n([1, 2, 3], 0), [])"]),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n"],
                   ["self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"],
                   ["self.assertEqual(m.count_even([]), 0)", "self.assertEqual(m.count_even([0, -2, 3]), 2)"]),
    "factorial": ("def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
                  ["def factorial(n):\n    result = 1\n    for i in range(1, n):\n        result *= i\n    return result\n"],
                  ["self.assertEqual(m.factorial(5), 120)"],
                  ["self.assertEqual(m.factorial(0), 1)", "self.assertEqual(m.factorial(6), 720)"]),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      ["def is_palindrome(s):\n    return s == ''.join(sorted(s))\n"],
                      ["self.assertTrue(m.is_palindrome('racecar'))", "self.assertFalse(m.is_palindrome('abc'))"],
                      ["self.assertTrue(m.is_palindrome('abba'))", "self.assertFalse(m.is_palindrome('ab'))",
                       "self.assertTrue(m.is_palindrome(''))"]),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
                     ["def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n"],
                     ["self.assertEqual(m.count_vowels('Apple'), 2)"],
                     ["self.assertEqual(m.count_vowels('xyz'), 0)", "self.assertEqual(m.count_vowels('AEIOU'), 5)"]),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n"],
              ["self.assertEqual(m.clamp(5, 0, 3), 3)", "self.assertEqual(m.clamp(-1, 0, 3), 0)"],
              ["self.assertEqual(m.clamp(2, 0, 3), 2)", "self.assertEqual(m.clamp(10, -5, 5), 5)"]),
}


def _test_file(cases: dict[str, list[str]]) -> str:
    methods = "\n".join(f"    def test_{name}(self):\n" + "".join(f"        {line}\n" for line in lines)
                        for name, lines in cases.items())
    return f"import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n{methods}\n\nif __name__ == '__main__':\n    unittest.main()\n"


def _contradiction(check: str, mixed: bool) -> str:
    """Проверка, противоречащая данной. mixed — в другой форме (assertTrue(x == y)), которую детектор без LLM не ловит."""
    node = ast.parse(check.strip(), mode="eval").body
    if node.func.attr in ("assertTrue", "assertFalse"):
        flipped = "assertFalse" if node.func.attr == "assertTrue" else "assertTrue"
        return f"self.{flipped}({ast.unparse(node.args[0])})"
    expr, value = ast.unparse(node.args[0]), ast.literal_eval(node.args[1])
    if isinstance(value, bool):
        other = not value
    elif isinstance(value, (int, float)):
        other = value + 1
    elif isinstance(value, str):
        other = value + "!"
    else:
        other = list(value) + [0]
    return f"self.assertTrue({expr} == {other!r})" if mixed else f"self.assertEqual({expr}, {other!r})"


def _tests_pass(module: str, tests: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "mylib.py").write_text(module, encoding="utf-8")
        Path(tmp, "test_mylib.py").write_text(tests, encoding="utf-8")
        proc = subprocess.run([sys.executable, "test_mylib.py"], cwd=tmp, capture_output=True, timeout=60)
        return proc.returncode == 0


def gen_coding(rng: random.Random, infeasible: str | None, paraphrase: bool) -> Generated:
    names = rng.sample(list(FUNCTIONS), rng.randint(3, 6))
    buggy = set(rng.sample(names, rng.randint(1, min(3, len(names)))))
    correct = "\n\n".join(FUNCTIONS[n][0] for n in names)
    module = "\n\n".join(rng.choice(FUNCTIONS[n][1]) if n in buggy else FUNCTIONS[n][0] for n in names)
    visible = {n: list(FUNCTIONS[n][2]) for n in names}
    hidden = {n: FUNCTIONS[n][2] + FUNCTIONS[n][3] for n in names}
    visible_tests, hidden_tests = _test_file(visible), _test_file(hidden)
    # Самопроверка: правильный код проходит все тесты, код с багами проваливает видимые.
    assert _tests_pass(correct, visible_tests) and _tests_pass(correct, hidden_tests), names
    assert not _tests_pass(module, visible_tests), names

    variant = "template"
    if infeasible:
        name = rng.choice(names)
        if infeasible == "impossible_assert":
            # f(x) != f(x) — чистая функция этого не выполнит никогда (только хак с состоянием).
            expr = ast.unparse(ast.parse(FUNCTIONS[name][2][0].strip(), mode="eval").body.args[0])
            visible[name].append(f"self.assertNotEqual({expr}, {expr})")
        else:
            visible[name].append(_contradiction(FUNCTIONS[name][2][0], mixed=infeasible == "contradiction_mixed"))
        visible_tests = _test_file(visible)
        assert not _tests_pass(correct, visible_tests), names
        variant = f"infeasible:{infeasible}"
    files = {"mylib.py": module, "test_mylib.py": visible_tests}
    return Generated("coding", CODING_INSTRUCTION, None if infeasible else "tests_pass", "unittest", variant,
                     files, {} if infeasible else {"test_hidden.py": hidden_tests})


# ---------------------------------------------------------------- сборка набора

GENERATORS = {"math": gen_math, "data_analysis": gen_data_analysis, "search": gen_search, "coding": gen_coding}
INFEASIBLE_KINDS = {
    "math": ["unknown_start", "vague_quantity", "zero_split", "missing_duration", "missing_parts"],
    "data_analysis": ["missing_column", "missing_file", "returns_not_tracked", "missing_column_phrasing"],
    "search": ["absent_person", "dangling_employer", "missing_mentor", "namesakes", "unknown_field",
               "absent_company", "ambiguous_first_name"],
    "coding": ["contradiction", "contradiction_mixed", "impossible_assert"],
}


def write_task(root: Path, task_id: str, gen: Generated, reward: int, seed: int) -> None:
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    data = {"task_id": task_id, "task_type": gen.task_type, "instruction": gen.instruction, "reward": reward,
            "expected": gen.expected, "check": gen.check}
    if not gen.feasible:
        data["feasible"] = False
    data["generator"] = {"variant": gen.variant, "seed": seed}
    (task_dir / "task.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for rel, text in gen.files.items():
        path = task_dir / "workspace" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for rel, text in gen.hidden.items():
        (task_dir / "hidden").mkdir(exist_ok=True)
        (task_dir / "hidden" / rel).write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("tasks_generated"))
    parser.add_argument("--types", nargs="*", default=list(GENERATORS), choices=list(GENERATORS))
    parser.add_argument("--infeasible-share", type=float, default=0.2, help="доля невыполнимых (в финале 15–25%%)")
    parser.add_argument("--paraphrase-share", type=float, default=0.3, help="доля перефразировок среди выполнимых")
    parser.add_argument("--infeasible-kinds", nargs="*", default=None,
                        help="только эти виды невыполнимых задач (по умолчанию все из INFEASIBLE_KINDS)")
    parser.add_argument("--force", action="store_true", help="перезаписать существующую папку")
    args = parser.parse_args()

    if args.out.exists():
        if not args.force:
            print(f"{args.out} уже существует; добавьте --force, чтобы перезаписать", file=sys.stderr)
            return 2
        shutil.rmtree(args.out)
    rng = random.Random(args.seed)
    n_infeasible = round(args.count * args.infeasible_share)
    kinds = {t: [k for k in INFEASIBLE_KINDS[t] if args.infeasible_kinds is None or k in args.infeasible_kinds]
             for t in args.types}
    infeasible_types = [t for t in args.types if kinds[t]]
    if n_infeasible and not infeasible_types:
        parser.error("для выбранных типов нет таких видов невыполнимых задач")
    plan = [(infeasible_types[i % len(infeasible_types)], True) if i < n_infeasible
            else (args.types[i % len(args.types)], False) for i in range(args.count)]
    rng.shuffle(plan)

    stats: dict[str, int] = {}
    for index, (task_type, infeasible) in enumerate(plan):
        kind = rng.choice(kinds[task_type]) if infeasible else None
        gen = GENERATORS[task_type](rng, kind, paraphrase=rng.random() < args.paraphrase_share)
        write_task(args.out, f"{task_type}_{index:04d}", gen, rng.randint(1, 5), args.seed)
        key = f"{task_type}/{gen.variant}"
        stats[key] = stats.get(key, 0) + 1

    print(f"generated {args.count} tasks in {args.out} ({n_infeasible} infeasible)")
    for key in sorted(stats):
        print(f"  {key:<45} {stats[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
