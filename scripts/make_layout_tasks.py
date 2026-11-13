"""Задачи с раскладкой, которую решение одним вызовом не берёт: их решает цикл агента (ReActAgent + bash).

    python scripts/make_layout_tasks.py tasks_generated_layout
    python scripts/run_dev.py --tasks tasks_generated_layout

Два CSV (нужно соединить), JSON вместо CSV, длинная база знаний связным текстом (> oneshot.MAX_KB_CHARS),
coding с двумя модулями. В task.json есть expected — для scripts/evaluate.py.
"""
import json
import random
import shutil
import sys
from pathlib import Path

TAIL = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODING = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
          "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are done, "
          "write ONLY the word DONE to the file answer.txt in the current directory (for example with bash: "
          "echo 42 > answer.txt).")
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo"]
FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya", "Nikolai",
         "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Garcia", "Haddad", "Ivanova", "Kowalski", "Muller", "Novak", "Petrov", "Sato", "Smith", "Wang"]
CITIES = ["Oslo", "Lisbon", "Osaka", "Berlin", "Toronto", "Seoul", "Austin", "Lyon"]
HOBBIES = ["chess", "sailing", "painting", "climbing", "the violin", "gardening", "astronomy", "pottery"]


def write(root: Path, task_id: str, task_type: str, instruction: str, expected: str, check: str,
          files: dict[str, str], hidden: dict[str, str] | None = None) -> None:
    task = root / task_id
    (task / "workspace").mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({
        "task_id": task_id, "task_type": task_type, "instruction": instruction, "reward": 3,
        "expected": expected, "check": check, "generator": {"variant": f"layout:{task_id.split('_', 1)[1]}"},
    }, indent=2) + "\n", encoding="utf-8")
    for rel, text in files.items():
        path = task / "workspace" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for rel, text in (hidden or {}).items():
        (task / "hidden").mkdir(exist_ok=True)
        (task / "hidden" / rel).write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(sys.argv[1])
    shutil.rmtree(root, ignore_errors=True)
    rng = random.Random(7)
    for n in range(3):  # два CSV: заказы + справочник товаров
        prices = {i + 1: (p, rng.randint(5, 60)) for i, p in enumerate(PRODUCTS)}
        orders = [(k + 1, rng.randint(1, 4), rng.randint(1, 20), rng.choice(["north", "south"])) for k in range(60)]
        product = rng.choice(PRODUCTS)
        pid = next(i for i, (p, _) in prices.items() if p == product)
        answer = sum(u * prices[pid][1] for _, p, u, _ in orders if p == pid)
        write(root, f"data_analysis_join{n}", "data_analysis",
              "The files orders.csv (columns order_id,product_id,units,region) and products.csv (columns "
              f"product_id,name,unit_price) describe sales. What was the total revenue (units times unit_price) from "
              f"product '{product}'? Use bash with python3 to compute it. " + TAIL, str(answer), "exact_int",
              {"orders.csv": "order_id,product_id,units,region\n" + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in orders),
               "products.csv": "product_id,name,unit_price\n" + "".join(f"{i},{p},{c}\n" for i, (p, c) in prices.items())})
    for n in range(3):  # JSON вместо CSV
        records = [{"id": k + 1, "product": rng.choice(PRODUCTS), "month": rng.randint(1, 12), "units": rng.randint(1, 30)}
                   for k in range(50)]
        product, month = rng.choice(PRODUCTS), rng.randint(1, 12)
        answer = sum(r["units"] for r in records if r["product"] == product and r["month"] <= month)
        write(root, f"data_analysis_json{n}", "data_analysis",
              f"The file sales.json holds a list of sales records with fields id, product, month and units. How many "
              f"units of '{product}' were sold in months 1 through {month} inclusive? Use bash with python3 to compute "
              "it. " + TAIL, str(answer), "exact_int", {"sales.json": json.dumps(records, indent=1)})
    for n in range(3):  # длинная база знаний связным текстом
        names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 60)
        people = {p: (rng.randint(1950, 2000), rng.choice(CITIES), rng.choice(HOBBIES)) for p in names}
        person = rng.choice(names)
        files = {f"kb/{p.lower().replace(' ', '_')}.md":
                 f"# {p}\n\n{p} was born in {y} and grew up in a small town before moving to {c}, where {p.split()[0]} "
                 f"still lives today. Friends describe {p.split()[0]} as patient and curious. In the evenings "
                 f"{p.split()[0]} enjoys {h}, and on weekends often visits the old harbour with family. "
                 f"{p.split()[0]} has worked in several companies over the years and speaks three languages.\n"
                 for p, (y, c, h) in people.items()}
        write(root, f"search_longkb{n}", "search",
              f"The directory kb/ contains a knowledge base of biographies. In which city does {person} live? Use the "
              "file tools or bash (ls, grep, cat) to find it. " + TAIL, people[person][1], "exact_str", files)
    for n in range(2):  # coding с двумя модулями: решение одним вызовом не берёт
        helpers = "def clean(words):\n    return [w.strip().lower() for w in words if w.strip()]\n"
        buggy = ("from helpers import clean\n\n\ndef count_words(text):\n    return len(clean(text.split(',')))\n\n\n"
                 "def longest(words):\n    return min(clean(words), key=len)\n")
        tests = ("import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
                 "    def test_count(self):\n        self.assertEqual(m.count_words('a b  c'), 3)\n\n"
                 "    def test_longest(self):\n        self.assertEqual(m.longest(['ab', ' abcd ', 'a']), 'abcd')\n\n\n"
                 "if __name__ == '__main__':\n    unittest.main()\n")
        hidden = tests.replace("'a b  c'), 3)", "'one two'), 2)\n        self.assertEqual(m.count_words(''), 0)")
        write(root, f"coding_twomodules{n}", "coding", CODING, "tests_pass", "unittest",
              {"mylib.py": buggy, "helpers.py": helpers, "test_mylib.py": tests}, {"test_hidden.py": hidden})
    print(f"written {len(list(root.iterdir()))} tasks to {root}")


main()
