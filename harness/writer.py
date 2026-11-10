"""Запись ответов. Вызывается сразу после каждой задачи: несохранённое не засчитывается."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from harness.tasks import Task

log = logging.getLogger(__name__)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class AnswerWriter:
    def __init__(self, out_file: Path | None):
        self._out_file = out_file
        self._solutions: dict[str, str] = {}

    def save(self, task: Task, answer: str) -> None:
        # Организаторы: answer.txt кладём рядом с task.json. Для невыполнимой задачи файл не создаём,
        # поэтому save() вызывается только когда ответ есть.
        path = task.task_dir / "answer.txt"
        try:
            write_atomic(path, answer)
        except OSError as exc:
            log.error("[%s] cannot write %s: %s", task.task_id, path, exc)

        self._solutions[task.task_id] = answer
        if self._out_file is not None:
            try:
                write_atomic(self._out_file, json.dumps(self._solutions, ensure_ascii=False, indent=1))
            except OSError as exc:
                log.error("cannot write solutions file %s: %s", self._out_file, exc)
