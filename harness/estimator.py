"""Оценщик стоимости задачи до запуска: сколько токенов (out + in/10) потратит наш харнесс.

Работает только по task.json и файлам задачи, без LLM и без запуска кода задачи.
Два уровня:
  1. Маршрут — теми же проверками, что у харнесса: быстрый путь (0 токенов), противоречивые тесты coding (0),
     coding или другой тип за один вызов (oneshot), цикл агента по типу задачи.
  2. Внутри маршрута — линейная модель по статическим признакам (параметры — harness/estimator_params.json,
     их подбирает scripts/fit_estimator.py по траекториям). Без файла параметров — априорные средние.
Кроме точечной оценки даём верхнюю (upper): организаторы обещали штраф за недооценку.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from harness import fastpath
from harness.coding import MAX_SOURCE_BYTES, contradictory_tests
from harness.oneshot import supports_oneshot
from harness.tasks import Task

PARAMS_PATH = Path(__file__).with_name("estimator_params.json")
# Априорные средние по маршрутам (замеры на qwen3.8-27b-noreason) — пока нет подобранных параметров.
DEFAULT_COST = {"coding": 110, "oneshot:math": 70, "oneshot:data_analysis": 100, "oneshot:search": 100,
                "agent:data_analysis": 300, "agent:math": 220, "agent:search": 480}
DEFAULT_AGENT_COST = 400
DEFAULT_UPPER_FACTOR = 2.0
MIN_LLM_COST = 20


@dataclass
class Estimate:
    task_id: str
    route: str
    cost: float
    upper: float
    features: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "route": self.route, "estimate": round(self.cost),
                "upper": round(self.upper)}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_features(task: Task) -> dict[str, float]:
    """Статические признаки задачи; одинаковые для обучения и прогноза."""
    files = [p for p in task.workspace.rglob("*") if p.is_file() and "__pycache__" not in p.parts] \
        if task.workspace else []
    features = {
        "instruction_chars": float(len(task.instruction)),
        "n_files": float(len(files)),
        "workspace_kb": min(sum(p.stat().st_size for p in files), 500_000) / 1024,
    }
    if task.task_type == "coding" and task.workspace:
        modules = [p for p in files if p.suffix == ".py" and not p.name.startswith("test_")]
        tests = [p for p in files if p.suffix == ".py" and p.name.startswith("test_")]
        module_text = "".join(_read(p) for p in modules)
        tests_text = "".join(_read(p) for p in tests)
        features.update(
            module_chars=float(len(module_text)),
            tests_chars=float(len(tests_text)),
            n_functions=float(sum(1 for line in module_text.splitlines() if line.startswith("def "))),
            n_asserts=float(tests_text.count("self.assert")),
        )
    if task.task_type == "data_analysis":
        features["csv_rows"] = float(sum(max(len(_read(p).splitlines()) - 1, 0) for p in files if p.suffix == ".csv"))
    return features


def detect_route(task: Task) -> str:
    """По какому пути харнесс будет решать задачу (повторяет логику runner/solver/coding)."""
    if fastpath.try_solve(task, task.workspace or task.task_dir) is not None:
        return "fastpath"
    if task.task_type == "coding" and task.workspace:
        tests = sorted(task.workspace.glob("test_*.py"))
        modules = [p for p in task.workspace.glob("*.py") if not p.name.startswith("test_")]
        if tests and len(modules) == 1 and modules[0].stat().st_size <= MAX_SOURCE_BYTES:
            if contradictory_tests([_read(t) for t in tests]):
                return "coding:contradiction"
            return "coding"
    if supports_oneshot(task):
        return f"oneshot:{task.task_type}"
    return f"agent:{task.task_type}"


def load_params(path: Path = PARAMS_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("routes", {})
    except (OSError, ValueError):
        return {}


def predict(route: str, features: dict[str, float], params: dict) -> tuple[float, float]:
    """(точечная оценка, верхняя оценка) для маршрута."""
    if route in ("fastpath", "coding:contradiction"):
        return 0.0, 0.0
    model = params.get(route)
    if model is None:
        cost = DEFAULT_COST.get(route, DEFAULT_AGENT_COST)
        return float(cost), cost * DEFAULT_UPPER_FACTOR
    cost = model["intercept"] + sum(c * features.get(name, 0.0) for name, c in model.get("coef", {}).items())
    cost = max(cost, MIN_LLM_COST)
    return cost, cost * model.get("upper_factor", DEFAULT_UPPER_FACTOR)


def estimate(task: Task, params: dict | None = None, route: str | None = None) -> Estimate:
    params = load_params() if params is None else params
    route = route or detect_route(task)
    features = extract_features(task)
    cost, upper = predict(route, features, params)
    return Estimate(task.task_id, route, cost, upper, features)
