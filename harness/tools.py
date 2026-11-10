"""Инструменты агента: bash в рабочей папке задачи."""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import uuid
from pathlib import Path

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard

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


def make_bash_tool(task_id: str, workdir: Path, timeout_s: float, max_output: int) -> LocalFunction:
    card = ToolCard(
        # id должен быть уникальным: инструменты регистрируются в общем Runner.
        id=f"bash_{task_id}_{uuid.uuid4().hex[:8]}",
        name="bash",
        description="Run a bash command in the task directory. Returns exit code and output.",
        input_params={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "bash command to run"}},
            "required": ["command"],
        },
    )

    async def bash(command: str) -> str:
        code, output = await run_command(command, workdir, timeout_s, max_output)
        return f"exit_code={code}\n{output}"

    return LocalFunction(card=card, func=bash)
