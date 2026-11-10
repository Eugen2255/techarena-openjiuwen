"""Инструменты агента: bash в рабочей папке задачи."""
from __future__ import annotations

import uuid
from pathlib import Path

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard

from harness.shell import run_command


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
