"""Детерминированные решатели — без LLM и без токенов.

Организаторы разрешили решать задачи без модели («можно решить все, если получится»),
а при равенстве баллов выигрывает тот, кто потратил меньше токенов.

Правило для решателя: ответ возвращается, только если шаблон распознан целиком и данные проверены.
Любое сомнение — вернуть None, и задача уйдёт агенту. Неверный ответ отсюда никто не перепроверит.

Новый решатель: модуль в этом пакете с функцией под декоратором @register("<task_type>"),
плюс импорт модуля внизу этого файла.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from harness.tasks import Task

log = logging.getLogger(__name__)


@dataclass
class FastAnswer:
    answer: str | None          # None вместе с infeasible=True: задачу не решаем, answer.txt не создаём
    infeasible: bool = False
    note: str = ""


Solver = Callable[[Task, Path], "FastAnswer | None"]
_SOLVERS: dict[str, list[Solver]] = {}


def register(task_type: str) -> Callable[[Solver], Solver]:
    def decorator(func: Solver) -> Solver:
        _SOLVERS.setdefault(task_type, []).append(func)
        return func
    return decorator


def try_solve(task: Task, workdir: Path) -> FastAnswer | None:
    for solver in _SOLVERS.get(task.task_type, []):
        try:
            result = solver(task, workdir)
        except Exception:  # noqa: BLE001 — сломанный решатель не должен ронять прогон
            log.exception("[%s] fastpath %s failed", task.task_id, solver.__name__)
            continue
        if result is not None:
            return result
    return None


# Регистрация решателей (импорт ради побочного эффекта @register).
from harness.fastpath import data_analysis, math_tasks, search  # noqa: E402,F401
