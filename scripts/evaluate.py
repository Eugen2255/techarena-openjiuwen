#!/usr/bin/env python3
"""Оценка ответов в каталоге задач по полям expected/check.

Эти поля есть только в примерах и в наших сгенерированных задачах. Невыполнимые задачи в наших наборах
помечены `"feasible": false` и `"expected": null`: правильно — не создавать answer.txt (баллов это не даёт).
Для coding, кроме видимых тестов, запускаются скрытые из <task>/hidden/ (их кладёт scripts/generate_tasks.py).

    python scripts/evaluate.py runs/<время>/tasks --original tasks_examples --logs runs/<время>/logs [--all]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def read_answer(task_dir: Path) -> str | None:
    """Организаторы читают answer.txt рядом с task.json."""
    path = task_dir / "answer.txt"
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else None


def tests_pass(workspace: Path, original: Path | None) -> tuple[bool, str]:
    tests = sorted(workspace.glob("test_*.py"))
    if not tests:
        return False, "no tests"
    for test in tests:
        if original is not None:
            orig = original / test.name
            if orig.is_file() and orig.read_bytes() != test.read_bytes():
                return False, f"{test.name} was modified"
        try:
            proc = subprocess.run([sys.executable, test.name], cwd=workspace, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, f"{test.name} timeout"
        if proc.returncode != 0:
            return False, f"{test.name} fails"
    return True, ""


def hidden_tests_pass(task_dir: Path, original: Path | None) -> tuple[bool, str]:
    """Скрытые тесты лежат в исходном наборе (в прогон их не копируют); запускаем на копии workspace."""
    hidden_dir = (original / task_dir.name / "hidden") if original else (task_dir / "hidden")
    hidden = sorted(hidden_dir.glob("test_*.py"))
    if not hidden:
        return True, ""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(task_dir / "workspace", tmp, dirs_exist_ok=True)
        for test in hidden:
            shutil.copy(test, tmp)
            try:
                proc = subprocess.run([sys.executable, test.name], cwd=tmp, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                return False, f"hidden {test.name} timeout"
            if proc.returncode != 0:
                return False, f"hidden {test.name} fails"
    return True, ""


def check(task_dir: Path, data: dict, answer: str | None, original: Path | None) -> tuple[bool, str]:
    if data.get("feasible") is False:
        return answer is None, "infeasible: skipped" if answer is None else "infeasible, but answer.txt created"
    if answer is None:
        return False, "no answer"
    kind, expected = data.get("check"), data.get("expected")
    if kind == "unittest":
        orig_ws = original / task_dir.name / "workspace" if original else None
        ok, note = tests_pass(task_dir / "workspace", orig_ws)
        return hidden_tests_pass(task_dir, original) if ok else (ok, note)
    if expected is None:
        return False, "no expected"
    if kind == "exact_int":
        try:
            return int(answer) == int(expected), ""
        except ValueError:
            return False, "not an int"
    return answer == str(expected).strip(), ""


def load_costs(logs_dir: Path | None) -> dict[str, float]:
    costs: dict[str, float] = {}
    if logs_dir is None or not logs_dir.is_dir():
        return costs
    for path in sorted(logs_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                costs[rec["task_id"]] = costs.get(rec["task_id"], 0) + float(rec.get("cost", 0))
    return costs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("--original", type=Path, help="исходный набор, чтобы проверить, что тесты не изменены")
    parser.add_argument("--logs", type=Path, help="папка с trajectories_*.jsonl")
    parser.add_argument("--all", action="store_true", help="построчно все задачи (по умолчанию — только ошибки, "
                                                           "если задач больше 20)")
    args = parser.parse_args()

    costs = load_costs(args.logs)
    task_dirs = sorted(p for p in args.tasks_dir.iterdir() if (p / "task.json").is_file())
    show_all = args.all or len(task_dirs) <= 20
    score = max_score = solved = total = 0
    skipped_ok = infeasible_total = 0
    groups: dict[str, list] = defaultdict(lambda: [0, 0, 0.0])  # вид задачи -> [верно, всего, токены]
    print(f"{'task':<22} {'type':<14} {'rew':>3} {'ok':<3} {'cost':>7}  answer / note")
    for task_dir in task_dirs:
        data = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        answer = read_answer(task_dir)
        ok, note = check(task_dir, data, answer, args.original)
        reward = float(data.get("reward") or 0)
        if data.get("feasible") is False:
            infeasible_total += 1
            skipped_ok += ok
        else:
            total += 1
            max_score += reward
            if ok:
                solved += 1
                score += reward
        cost = costs.get(data.get("task_id", task_dir.name))
        group = groups[f"{data.get('task_type', '?')}/{data.get('generator', {}).get('variant', '-')}"]
        group[0] += ok
        group[1] += 1
        group[2] += cost or 0
        if show_all or not ok:
            cost_s = f"{cost:7.0f}" if cost is not None else "      -"
            shown = (answer or "")[:30]
            print(f"{task_dir.name:<22} {data.get('task_type', '?'):<14} {reward:>3.0f} {'+' if ok else '-':<3} "
                  f"{cost_s}  {shown!r} {note}")

    if len(groups) > 1:
        print(f"\n{'type/variant':<45} {'ok':>9} {'tokens':>8}")
        for key in sorted(groups):
            right, count, tokens = groups[key]
            print(f"{key:<45} {right:>4}/{count:<4} {tokens:>8.0f}")
    total_cost = sum(costs.values())
    print(f"\nsolved {solved}/{total}, score {score:.0f}/{max_score:.0f}, "
          f"tokens (out + in/10) {total_cost:.0f}")
    if infeasible_total:
        print(f"infeasible correctly skipped {skipped_ok}/{infeasible_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
