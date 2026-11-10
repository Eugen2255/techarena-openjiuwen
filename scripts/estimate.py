#!/usr/bin/env python3
"""Прогноз стоимости задач до запуска (без LLM): сколько токенов (out + in/10) потратит харнесс.

    python scripts/estimate.py tasks_generated                 # таблица и итог
    python scripts/estimate.py /data/tasks --json predictions.json

Организаторы будут оценивать прогноз на новых задачах; формат сдачи и метрики пока не опубликованы,
поэтому выдаём и точечную оценку (estimate), и верхнюю (upper — на случай штрафа за недооценку).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import estimator  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("--json", type=Path, help="сохранить прогнозы в JSON")
    args = parser.parse_args()

    params = estimator.load_params()
    predictions = [estimator.estimate(task, params) for task in load_tasks(args.tasks_dir)]
    routes = Counter(p.route for p in predictions)
    for p in predictions:
        print(f"{p.task_id:<24} {p.route:<24} {p.cost:>7.0f} {p.upper:>7.0f}")
    print(f"\n{len(predictions)} задач; маршруты: {dict(sorted(routes.items()))}")
    print(f"прогноз на весь набор: {sum(p.cost for p in predictions):.0f} токенов "
          f"(верхний {sum(p.upper for p in predictions):.0f}); параметры: "
          f"{'harness/estimator_params.json' if params else 'априорные средние'}")
    if args.json:
        args.json.write_text(json.dumps([p.as_dict() for p in predictions], indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
