#!/usr/bin/env python3
"""Подбор параметров оценщика стоимости (harness/estimator.py) по траекториям прогонов.

    python scripts/fit_estimator.py runs/<прогон1> runs/<прогон2> --holdout runs/<прогон3> --write
    python scripts/fit_estimator.py runs/<прогон>... --export trajectories/estimator_dataset.jsonl
    python scripts/fit_estimator.py --dataset trajectories/estimator_dataset.jsonl --write   # без прогонов

Каждый прогон — папка от scripts/run_dev.py: logs/trajectories_*.jsonl и source.txt (путь к исходному набору;
признаки считаем по исходным задачам: в папке прогона код coding уже исправлен моделью).
--export сохраняет выборку (маршрут, признаки, факт) — она лежит в репозитории, и по ней можно дообучать оценщик.
На каждом маршруте выбираем модель по ошибке leave-one-out: среднее, один признак или все признаки (ridge).
Верхняя оценка = точечная × 90-й процентиль отношения факт/прогноз на LOO.
--write сохраняет harness/estimator_params.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness import estimator  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402

ROUTE_FEATURES = {"coding": ["module_chars", "n_asserts", "n_functions"]}
DEFAULT_FEATURES = ["instruction_chars", "n_files", "workspace_kb"]
RIDGE = 1.0
UPPER_QUANTILE = 0.9


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Гаусс с выбором главного элемента; матрица маленькая (до 3×3)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_linear(rows: list[list[float]], y: list[float], ridge: float = RIDGE) -> tuple[float, list[float]]:
    """Ridge-регрессия на стандартизованных признаках; свободный член без штрафа."""
    if not rows or not rows[0]:
        return statistics.fmean(y), []
    cols = list(zip(*rows))
    means = [statistics.fmean(c) for c in cols]
    stds = [statistics.pstdev(c) or 1.0 for c in cols]
    z = [[(x - mu) / sd for x, mu, sd in zip(row, means, stds)] for row in rows]
    y_mean = statistics.fmean(y)
    k = len(means)
    a = [[sum(r[i] * r[j] for r in z) + (ridge if i == j else 0.0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * (t - y_mean) for r, t in zip(z, y)) for i in range(k)]
    w = solve_linear(a, b)
    coef = [wi / sd for wi, sd in zip(w, stds)]
    return y_mean - sum(c * mu for c, mu in zip(coef, means)), coef


def loo_predictions(rows: list[list[float]], y: list[float]) -> list[float]:
    preds = []
    for i in range(len(y)):
        intercept, coef = fit_linear(rows[:i] + rows[i + 1:], y[:i] + y[i + 1:])
        preds.append(max(intercept + sum(c * x for c, x in zip(coef, rows[i])), estimator.MIN_LLM_COST))
    return preds


def metrics(actual: list[float], predicted: list[float]) -> dict:
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    return {"mae": statistics.fmean(errors),
            "mape": statistics.fmean(e / a for e, a in zip(errors, actual) if a > 0) if any(actual) else 0.0,
            "under": sum(p < a for a, p in zip(actual, predicted)) / len(actual)}


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def consistent(solver: str, route: str, cost: float) -> bool:
    """Запись показательна, только если харнесс и сейчас решал бы задачу тем же путём."""
    if solver == "fastpath":
        return route == "fastpath"
    if solver == "coding":
        # Противоречивые тесты теперь отсеиваются без LLM; старые записи с расходом устарели.
        return route == "coding" or (route == "coding:contradiction" and cost == 0)
    if solver == "oneshot":
        return route.startswith("oneshot:")
    return solver == "llm" and route.startswith("agent:")


def group(records: list[dict]) -> tuple[dict[str, list[tuple[dict, float]]], list[tuple[str, dict, float]]]:
    """(признаки, факт) по маршрутам с вызовами LLM и полный список (маршрут, признаки, факт)."""
    by_route: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    everything = []
    for rec in records:
        everything.append((rec["route"], rec["features"], rec["cost"]))
        if rec["route"] not in ("fastpath", "coding:contradiction"):
            by_route[rec["route"]].append((rec["features"], rec["cost"]))
    return by_route, everything


def read_dataset(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_records(run_dirs: list[Path]) -> list[dict]:
    """Записи выборки из прогонов: задача, маршрут, признаки (по исходной задаче), фактический расход."""
    records = []
    for run in run_dirs:
        source = Path((run / "source.txt").read_text(encoding="utf-8").strip())
        tasks = {t.task_id: t for t in load_tasks(source)}
        for line in "".join(p.read_text(encoding="utf-8") for p in sorted((run / "logs").glob("*.jsonl"))).splitlines():
            rec = json.loads(line)
            task = tasks.get(rec["task_id"])
            if task is None or rec.get("error"):
                continue
            route = estimator.detect_route(task)
            if not consistent(rec["solver"], route, float(rec["cost"])):
                continue  # маршрут изменился с тех пор (например, задачу теперь берёт быстрый путь)
            records.append({"run": run.name, "task_id": task.task_id, "task_type": task.task_type,
                            "variant": task.raw.get("generator", {}).get("variant"), "route": route,
                            "features": estimator.extract_features(task), "cost": float(rec["cost"])})
    return records


def fit_route(route: str, samples: list[tuple[dict, float]]) -> dict:
    names = ROUTE_FEATURES.get(route, DEFAULT_FEATURES)
    y = [cost for _, cost in samples]
    candidates = [[]] + ([[n] for n in names] + [names] if len(samples) >= 8 else [])
    best = None
    for chosen in candidates:
        rows = [[f.get(n, 0.0) for n in chosen] for f, _ in samples]
        preds = loo_predictions(rows, y) if len(samples) > 1 else [y[0]]
        score = metrics(y, preds)["mae"]
        if best is None or score < best[0] - 1e-9:
            best = (score, chosen, rows, preds)
    _, chosen, rows, preds = best
    intercept, coef = fit_linear(rows, y)
    ratios = [a / p for a, p in zip(y, preds) if p > 0]
    upper_factor = max(1.0, quantile(ratios, UPPER_QUANTILE)) if len(ratios) >= 3 else estimator.DEFAULT_UPPER_FACTOR
    loo = metrics(y, preds)
    loo_upper = metrics(y, [p * upper_factor for p in preds])
    return {"intercept": round(intercept, 4), "coef": {n: round(c, 6) for n, c in zip(chosen, coef)},
            "upper_factor": round(upper_factor, 3), "n": len(samples), "mean_cost": round(statistics.fmean(y), 1),
            "loo_mae": round(loo["mae"], 1), "loo_mape": round(loo["mape"], 3), "loo_under": round(loo["under"], 3),
            "loo_upper_under": round(loo_upper["under"], 3)}


def report(title: str, everything: list[tuple[str, dict, float]], params: dict) -> None:
    actual = [cost for _, _, cost in everything]
    point, upper = zip(*(estimator.predict(route, f, params) for route, f, _ in everything))
    llm = [i for i, (route, _, _) in enumerate(everything) if route not in ("fastpath", "coding:contradiction")]
    print(f"\n{title}: {len(everything)} задач, из них {len(llm)} с вызовами LLM; "
          f"факт {sum(actual):.0f} токенов, прогноз {sum(point):.0f}, верхний {sum(upper):.0f}")
    for name, idx in (("все задачи", range(len(actual))), ("только LLM", llm)):
        if not idx:
            continue
        a = [actual[i] for i in idx]
        for kind, pred in (("точечный", point), ("верхний", upper)):
            m = metrics(a, [pred[i] for i in idx])
            print(f"  {name:<11} {kind:<9} MAE {m['mae']:6.1f}  MAPE {m['mape']:6.1%}  недооценка {m['under']:6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="прогоны для обучения")
    parser.add_argument("--dataset", nargs="*", type=Path, default=[], help="готовая выборка (jsonl от --export)")
    parser.add_argument("--holdout", nargs="*", type=Path, default=[], help="прогоны только для проверки")
    parser.add_argument("--export", type=Path, help="сохранить обучающую выборку в jsonl")
    parser.add_argument("--write", action="store_true", help="сохранить harness/estimator_params.json")
    args = parser.parse_args()

    records = load_records(args.runs) + read_dataset(args.dataset)
    if not records:
        parser.error("нужны прогоны или --dataset")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        print(f"выборка ({len(records)} записей) сохранена в {args.export}")
    by_route, everything = group(records)
    params = {route: fit_route(route, samples) for route, samples in sorted(by_route.items())}
    print(f"{'маршрут':<24} {'n':>4} {'среднее':>8} {'LOO MAE':>8} {'MAPE':>7} {'недооц.':>8} {'upper×':>7}  признаки")
    for route, p in params.items():
        print(f"{route:<24} {p['n']:>4} {p['mean_cost']:>8.0f} {p['loo_mae']:>8.1f} {p['loo_mape']:>7.1%} "
              f"{p['loo_under']:>8.1%} {p['upper_factor']:>7.2f}  {', '.join(p['coef']) or 'среднее'}")
    report("Обучающие прогоны (в выборке)", everything, params)
    if args.holdout:
        report("Отложенные прогоны (вне выборки)", group(load_records(args.holdout))[1], params)
    if args.write:
        sources = [r.name for r in args.runs] + [p.as_posix() for p in args.dataset]
        estimator.PARAMS_PATH.write_text(json.dumps(
            {"fitted_at": time.strftime("%Y-%m-%d %H:%M"), "fitted_on": sources, "n_records": len(records),
             "routes": params}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nзаписано в {estimator.PARAMS_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
