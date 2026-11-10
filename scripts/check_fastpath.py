#!/usr/bin/env python3
"""Проверка решателей быстрого пути на наборе с expected — без LLM, Docker и токенов.

    python scripts/check_fastpath.py tasks_generated
    python scripts/check_fastpath.py tasks_examples

Для каждой задачи: взял ли её быстрый путь, верен ли ответ, верно ли объявлена невыполнимость.
Главное правило: ошибок быть не должно (WRONG, FALSE_INFEASIBLE, ANSWERED_INFEASIBLE = 0) — ответы быстрого пути
никто не перепроверяет. Код возврата 1, если ошибки есть.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.fastpath import try_solve  # noqa: E402
from harness.tasks import Task, load_tasks  # noqa: E402

ERRORS = ("WRONG", "FALSE_INFEASIBLE", "ANSWERED_INFEASIBLE")


def classify(task: Task) -> tuple[str, str]:
    feasible = task.raw.get("feasible") is not False
    result = try_solve(task, task.workspace or task.task_dir)
    if result is None:
        return "agent", ""
    if result.infeasible:
        return ("infeasible_ok", result.note) if not feasible else ("FALSE_INFEASIBLE", result.note)
    if not feasible:
        return "ANSWERED_INFEASIBLE", f"answer={result.answer!r}"
    expected = task.raw.get("expected")
    try:
        correct = (int(result.answer) == int(expected)) if task.check == "exact_int" else result.answer == str(expected)
    except (TypeError, ValueError):
        correct = False
    return ("solved", "") if correct else ("WRONG", f"answer={result.answer!r} expected={expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks_dir", type=Path)
    args = parser.parse_args()

    table: dict[str, Counter] = defaultdict(Counter)
    errors = []
    for task in load_tasks(args.tasks_dir):
        variant = task.raw.get("generator", {}).get("variant", "-")
        outcome, detail = classify(task)
        table[f"{task.task_type}/{variant}"][outcome] += 1
        if outcome in ERRORS:
            errors.append(f"{task.task_id} [{variant}] {outcome}: {detail}")

    columns = ["solved", "infeasible_ok", "agent", *ERRORS]
    print(f"{'type/variant':<42}" + "".join(f"{c:>20}" for c in columns))
    total = Counter()
    for key in sorted(table):
        total.update(table[key])
        print(f"{key:<42}" + "".join(f"{table[key][c]:>20}" for c in columns))
    print(f"{'TOTAL':<42}" + "".join(f"{total[c]:>20}" for c in columns))
    if errors:
        print("\nОшибки быстрого пути:")
        print("\n".join(f"  {e}" for e in errors))
        return 1
    print("\nОшибок нет: быстрый путь либо отвечает верно, либо отдаёт задачу агенту.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
