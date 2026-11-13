"""Загрузка задач из каталога tasks-dir."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    task_dir: Path
    task_type: str
    instruction: str
    reward: float
    check: str | None
    raw: dict = field(repr=False)

    @property
    def workspace(self) -> Path | None:
        """Папка workspace/ задачи. У задач math её нет."""
        ws = self.task_dir / "workspace"
        return ws if ws.is_dir() else None


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Читает все <tasks_dir>/*/task.json.

    Поле `expected` харнесс не использует: в финальном наборе его не будет,
    а на примерах это было бы подглядыванием в ответ.
    """
    tasks = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        task_file = task_dir / "task.json"
        if not task_file.is_file():
            continue
        try:
            data = json.loads(task_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            log.warning("skip %s: cannot read task.json (%s)", task_dir.name, exc)
            continue
        if not isinstance(data, dict):
            log.warning("skip %s: task.json is not an object", task_dir.name)
            continue
        tasks.append(Task(
            task_id=str(data.get("task_id") or task_dir.name),
            task_dir=task_dir,
            task_type=_task_type(data.get("task_type"), task_dir),
            instruction=str(data.get("instruction") or ""),
            reward=_reward(data.get("reward")),
            check=data.get("check") if isinstance(data.get("check"), str) else None,
            raw=data,
        ))
    return tasks


KNOWN_TYPES = ("math", "data_analysis", "search", "coding")


def _task_type(value, task_dir: Path) -> str:
    """«Data-Analysis», «data analysis» → «data_analysis»; поля нет — по имени папки (math_0004 → math)."""
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if text:
        return text
    guess = re.sub(r"_?\d+$", "", task_dir.name.lower())
    return guess if guess in KNOWN_TYPES else "unknown"


def _reward(value) -> float:
    """Награда числом; «3 points» → 3. Нет или не разобрать — 1 (обычная задача), а не падение всего прогона."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match[0]) if match else 1.0
