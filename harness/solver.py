"""Решение одной задачи агентом openJiuwen (ReActAgent + bash)."""
from __future__ import annotations

import logging
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

from harness.answers import INFEASIBLE, is_infeasible, normalize_answer
from harness.config import Settings
from harness.llm import client_config, make_model, request_config
from harness.meter import TokenMeter, Usage
from harness.oneshot import solve_oneshot
from harness.shell import run_command
from harness.tasks import Task
from harness.tools import make_bash_tool

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You solve one small task in a sandbox directory using the `bash` tool.
- Be brief. Prefer a single command that computes the result (e.g. python3 -c '...').
- Do not explore more than needed. Never edit test files. Work only inside the current directory.
- Do not create answer.txt: the harness saves your answer.
- If the task cannot be solved exactly with the given data, the answer is {INFEASIBLE}.
  Never assume a missing value (an unknown number, a missing file, column or record): that means {INFEASIBLE}.
- Your final reply is one line: ANSWER: <answer>"""

QUERY_FOOTER = "(Reply with `ANSWER: <answer>`; the harness writes answer.txt for you.)"

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)


@dataclass
class SolveResult:
    task_id: str
    task_type: str
    reward: float
    solver: str                      # "fastpath", "coding", "oneshot" или "llm" (цикл агента)
    model: str | None = None
    answer: str | None = None        # None — answer.txt не создаём
    infeasible: bool = False
    local_ok: bool | None = None
    error: str | None = None
    note: str = ""
    final_text: str = ""
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)

    def as_record(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "reward": self.reward,
            "solver": self.solver,
            "model": self.model,
            "answer": self.answer,
            "infeasible": self.infeasible,
            "local_ok": self.local_ok,
            "error": self.error,
            "note": self.note,
            "seconds": round(self.seconds, 1),
            **self.usage.as_dict(),
            "final_text": self.final_text[-500:],
        }


def workdir_for(task: Task) -> Path:
    """Папка, в которой работает агент. У задач без workspace — временная папка в /tmp."""
    if task.workspace is not None:
        return task.workspace
    scratch = Path(tempfile.gettempdir()) / "harness_work" / task.task_id
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def choose_model(task: Task, settings: Settings) -> str:
    """math — модель с рассуждениями (точнее на новых формулировках), остальное — без рассуждений (config.py)."""
    return settings.math_model if task.task_type == "math" else settings.model


async def gather_context(task: Task, workdir: Path, settings: Settings) -> str:
    """Контекст, собранный без LLM и подставленный в первый запрос.

    Главное место для экономии токенов: чем точнее контекст, тем меньше шагов агента.
    """
    files = sorted(p for p in workdir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    parts = []
    if files:
        shown = files[:60]
        listing = "\n".join(f"{p.relative_to(workdir).as_posix()} ({p.stat().st_size} B)" for p in shown)
        if len(files) > len(shown):
            listing += f"\n... and {len(files) - len(shown)} more files"
        parts.append(f"Files in the current directory:\n{listing}")

    if task.task_type == "coding":
        for path in files:
            if path.suffix == ".py" and path.stat().st_size < 8000:
                parts.append(f"--- {path.relative_to(workdir).as_posix()} ---\n"
                             f"{path.read_text(encoding='utf-8', errors='replace')}")
        for test in [p for p in files if p.name.startswith("test_") and p.suffix == ".py"][:2]:
            rel = test.relative_to(workdir).as_posix()
            code, output = await run_command(f"python3 {rel}", workdir, settings.command_timeout_s, 1500)
            parts.append(f"--- output of `python3 {rel}` (exit code {code}) ---\n{output}")

    if task.task_type == "data_analysis":
        for path in files:
            if path.suffix == ".csv":
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                head = "\n".join(lines[:3])
                parts.append(f"--- {path.name}: {max(len(lines) - 1, 0)} data rows, first lines ---\n{head}")

    return "\n\n".join(parts)


def _final_text(output) -> str:
    """Достаёт текст последнего ответа агента из результата Runner.run_agent."""
    if isinstance(output, dict):
        for key in ("output", "result", "content", "answer"):
            if key in output:
                return _final_text(output[key])
    for attr in ("result", "content", "output"):
        value = getattr(output, attr, None)
        if value is not None and value is not output:
            return _final_text(value)
    return "" if output is None else str(output)


def extract_answer(workdir: Path, final_text: str) -> str | None:
    """Ответ из строки `ANSWER: ...`; запасной вариант — answer.txt, если агент всё же его создал."""
    found = _ANSWER_RE.findall(final_text or "")
    if found:
        answer = found[-1].strip().strip("`").strip()
        if answer:
            return answer
    answer_file = workdir / "answer.txt"
    if answer_file.is_file():
        text = answer_file.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text
    return None


async def local_check(task: Task, workdir: Path, answer: str, settings: Settings) -> bool | None:
    """Проверка без LLM. None — проверить нечем."""
    if task.task_type == "coding" or task.check == "unittest":
        tests = sorted(workdir.glob("test_*.py"))
        if not tests:
            return None
        for test in tests:
            code, _ = await run_command(f"python3 {test.name}", workdir, settings.command_timeout_s, 500)
            if code != 0:
                return False
        return True
    if task.check == "exact_int":
        return re.fullmatch(r"-?\d+", answer) is not None
    return None


def _build_agent(task: Task, workdir: Path, settings: Settings, meter: TokenMeter, model_name: str):
    config = ReActAgentConfig(
        model_client_config=client_config(settings),
        model_config_obj=request_config(settings, model_name),
        prompt_template=[{"role": "system", "content": SYSTEM_PROMPT}],
        max_iterations=settings.max_steps,
    )
    agent = ReActAgent(card=AgentCard(id=f"agent_{task.task_id}", description="task solver"))
    agent.configure(config)
    agent.set_llm(make_model(settings, meter, task.task_id, model_name))

    tool = make_bash_tool(task.task_id, workdir, settings.command_timeout_s, settings.max_tool_output)
    Runner.resource_mgr.add_tool(tool)
    agent.ability_manager.add(tool.card)
    return agent, tool


def _unregister(tool) -> None:
    remove = getattr(Runner.resource_mgr, "remove_tool", None)
    if tool is None or remove is None:
        return
    try:
        remove(tool.card.id)
    except Exception:  # noqa: BLE001 — уборка не должна ронять решение
        log.debug("cannot unregister tool %s", tool.card.id, exc_info=True)


async def solve(task: Task, settings: Settings, meter: TokenMeter) -> SolveResult:
    started = time.monotonic()
    model_name = choose_model(task, settings)
    result = SolveResult(task.task_id, task.task_type, task.reward, solver="llm", model=model_name)
    workdir = workdir_for(task)
    # Официальный ответ лежит рядом с task.json; answer.txt, созданный агентом в workspace, убираем.
    stray_answer = workdir / "answer.txt"
    had_stray = stray_answer.exists()
    tool = None
    try:
        # Сначала один вызов (harness/oneshot.py, harness/coding.py); цикл агента — только если он не справился.
        outcome = await solve_oneshot(task, settings, meter, model_name)
        if outcome is not None:
            result.solver = "coding" if task.task_type == "coding" else "oneshot"
            result.answer, result.infeasible, result.local_ok = outcome.answer, outcome.infeasible, outcome.local_ok
            result.note, result.final_text = outcome.note, outcome.final_text
            return result
        context = await gather_context(task, workdir, settings)
        query = "\n\n".join(p for p in (task.instruction, context, QUERY_FOOTER) if p)
        agent, tool = _build_agent(task, workdir, settings, meter, model_name)
        output = await Runner.run_agent(
            agent=agent,
            inputs={"query": query, "conversation_id": f"{task.task_id}_{uuid.uuid4().hex[:8]}"},
        )
        result.final_text = _final_text(output)
        answer = extract_answer(workdir, result.final_text)
        if answer is not None:
            answer = normalize_answer(answer, task.check)
        if answer is not None and is_infeasible(answer):
            result.infeasible = True
        elif answer is not None:
            result.local_ok = await local_check(task, workdir, answer, settings)
            if result.local_ok is False and task.task_type == "coding":
                result.note = "tests still fail, answer withheld"
            else:
                result.answer = answer
    except Exception as exc:  # noqa: BLE001 — одна задача не должна ронять весь прогон
        result.error = f"{type(exc).__name__}: {exc}"
        # Полный трейсбек — только в DEBUG: при недоступном прокси он печатался бы на каждую задачу.
        log.error("[%s] solve failed: %s", task.task_id, result.error[:300],
                  exc_info=log.isEnabledFor(logging.DEBUG))
    finally:
        _unregister(tool)
        if not had_stray and stray_answer.is_file() and task.workspace is not None:
            stray_answer.unlink(missing_ok=True)
        result.seconds = time.monotonic() - started
        result.usage = meter.task_usage(task.task_id)
    return result
