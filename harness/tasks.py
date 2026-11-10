"""Загрузка задач из каталога tasks-dir."""
from __future__ import annotations

import json
import logging
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
            data = json.loads(task_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("skip %s: cannot read task.json (%s)", task_dir.name, exc)
            continue
        tasks.append(Task(
            task_id=str(data.get("task_id") or task_dir.name),
            task_dir=task_dir,
            task_type=str(data.get("task_type") or "unknown"),
            instruction=str(data.get("instruction") or ""),
            reward=float(data.get("reward") or 0),
            check=data.get("check"),
            raw=data,
        ))
    return tasks
