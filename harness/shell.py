"""Запуск shell-команд в папке задачи: таймаут, обрезка вывода, без секретов в окружении.

Без внешних зависимостей, чтобы решатели и тесты работали и без openJiuwen.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
from pathlib import Path

# Переменные, которые не должны попадать в команды агента (ключ прокси и т.п.).
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _command_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("LLM_") and not any(m in k.upper() for m in _SECRET_MARKERS)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _shell_argv(command: str) -> list[str]:
    shell = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
    return [shell, "-c", command]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[{len(text) - limit} chars truncated]...\n{text[-half:]}"


async def run_command(command: str, cwd: Path, timeout_s: float, max_output: int) -> tuple[int | None, str]:
    """Выполняет команду в cwd. Возвращает (код возврата или None при таймауте, вывод)."""
    proc = await asyncio.create_subprocess_exec(
        *_shell_argv(command),
        cwd=str(cwd),
        env=_command_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except asyncio.TimeoutError:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        await proc.wait()
        return None, f"[command timed out after {timeout_s:.0f}s]"
    return proc.returncode, _truncate(out.decode("utf-8", "replace"), max_output)
