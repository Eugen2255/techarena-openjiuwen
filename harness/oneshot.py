"""Один вызов LLM вместо цикла агента — для задач, которые не взял быстрый путь.

Цикл ReAct дорог: каждый шаг пересылает весь контекст заново. Здесь модель сразу получает всё нужное
и отвечает одним сообщением:
  math, data_analysis — короткой программой на Python; харнесс запускает её сам и берёт напечатанный ответ;
  search — строкой ANSWER по базе знаний, которую харнесс кладёт в запрос в сжатом виде;
  coding — своя схема в harness/coding.py.
Невыполнимость: модель отвечает INFEASIBLE; кроме того, харнесс сам видит обращение программы к колонке,
которой нет в файле (KeyError), и деление на 0 в math.
Не вышло (нет кода, программа дважды падает, ответа нет в базе) → None, задачу решает агент.
"""
from __future__ import annotations

import logging
import re
import shlex
import tempfile
import uuid
from pathlib import Path

from harness.answers import Outcome, is_infeasible, normalize_answer, question_text, says_infeasible
from harness.coding import solve_coding
from harness.config import Settings
from harness.meter import TokenMeter
from harness.shell import run_command
from harness.tasks import Task

log = logging.getLogger(__name__)

MAX_PROGRAM_ATTEMPTS = 2
MAX_KB_CHARS = 12000  # больше — базу целиком в запрос не кладём, решает агент
PREVIEW_ROWS = 3

MATH_PROMPT = """Translate the word problem into a short Python 3 program that prints only the final integer answer.
Follow the wording literally ("rounded down" is //, "doubled" is *2). Never invent missing numbers.
If the problem cannot be solved exactly with the given data (an unknown quantity, an undefined operation),
reply with the single word INFEASIBLE. Otherwise reply with one ```python block only."""

DATA_PROMPT = """Answer the question about the data file by writing a short Python 3 program (standard library only)
that reads the file from the current directory and prints only the final answer.
Compute exactly what the question asks. If it asks for a quantity that is neither a column of the file nor defined
in the question itself (e.g. "profit", "tax" or "rating" when there is no such column), or needs a file or value
the data does not have, reply with the single word INFEASIBLE. Never substitute a different quantity and never
print 0 for missing data. Read columns as row['name'], never .get() with a default.
Otherwise reply with one ```python block only."""

SEARCH_PROMPT = """Answer the question using only the knowledge base below. Follow references between entries
(an employer or a mentor has its own entry).
Reply with one line: ANSWER: <value exactly as written in the knowledge base>.
If the knowledge base does not contain enough information (a missing entry or field, or an ambiguous answer),
reply: ANSWER: INFEASIBLE"""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_KEY_ERROR_RE = re.compile(r"KeyError: ['\"](?P<key>[^'\"]+)['\"]")
_KB_DIR_RE = re.compile(r"The directory (?P<dir>[\w./-]+?)/? contains")
# «total discount», «average rating» — агрегат по величине; слова-счётчики («total number of units») не считаются.
_AGGREGATE_RE = re.compile(r"\b(?:total|sum of(?: the)?|average|mean)\s+(?P<word>[a-z]+)(?P<rest>[^?]*)", re.IGNORECASE)
_COUNT_WORDS = {"number", "amount", "count", "quantity", "sales", "orders", "rows", "records", "items",
                "transactions", "of"}


def missing_quantity(question: str, columns: list[str]) -> str | None:
    """Величина, которую вопрос агрегирует, но которой нет среди колонок и нет определения в самом вопросе.

    Модель в таких случаях склонна подменять величину другой (например, «скидку» — выручкой).
    """
    names = [c.lower() for c in columns]
    for match in _AGGREGATE_RE.finditer(question):
        word = match["word"].lower()
        stem = word.rstrip("s")
        if word in _COUNT_WORDS or any(n.startswith(stem) or stem.startswith(n) for n in names if n):
            continue
        if match["rest"].lstrip().startswith("(") or re.search(rf"\b{word}\b\s*(?:is|=|means|defined as)\b",
                                                                question, re.IGNORECASE):
            continue  # величина определена в вопросе: «total revenue (units times price)»
        return word
    return None


def data_files(task: Task) -> list[Path]:
    return sorted(task.workspace.rglob("*.csv")) if task.workspace else []


def kb_files(task: Task) -> list[Path]:
    if task.workspace is None:
        return []
    match = _KB_DIR_RE.search(task.instruction)
    kb_dir = task.workspace / (match["dir"] if match else "kb")
    return sorted(p for p in kb_dir.rglob("*") if p.is_file() and p.suffix in (".md", ".txt")) if kb_dir.is_dir() else []


def compact_kb(files: list[Path]) -> str:
    """Одна запись — одна строка: «Irina Smith | Born: 1983; Employer: Dune Robotics; Mentor: Alice Smith».

    Так модели проще следовать ссылкам между записями; по токенам выходит не дороже исходных файлов.
    """
    lines = []
    for path in files:
        parts = [line.strip().lstrip("#").strip() for line in path.read_text(encoding="utf-8", errors="replace")
                 .splitlines() if line.strip()]
        if parts:
            lines.append(parts[0] + (" | " + "; ".join(parts[1:]) if len(parts) > 1 else ""))
    return "\n".join(lines)


def supports_oneshot(task: Task) -> bool:
    """Подходит ли задача для решения одним вызовом (coding проверяет harness/coding.py сам)."""
    if task.task_type == "math":
        return True
    if task.task_type == "data_analysis":
        return len(data_files(task)) == 1
    if task.task_type == "search":
        files = kb_files(task)
        return bool(files) and len(compact_kb(files)) <= MAX_KB_CHARS
    return False


def extract_code(reply: str) -> str | None:
    code = "\n\n".join(_FENCE_RE.findall(reply)).strip()
    return code or None


def _last_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


async def _run_program(code: str, workdir: Path, settings: Settings) -> tuple[int | None, str]:
    folder = Path(tempfile.gettempdir()) / "harness_programs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.py"
    path.write_text(code, encoding="utf-8")
    try:
        return await run_command(f"python3 {shlex.quote(path.as_posix())}", workdir,
                                 settings.command_timeout_s, settings.max_tool_output)
    finally:
        path.unlink(missing_ok=True)


async def _solve_with_program(task: Task, settings: Settings, model) -> Outcome | None:
    header: list[str] = []
    user = question_text(task.instruction)
    if task.task_type == "data_analysis":
        workdir, data = task.workspace, data_files(task)[0]
        lines = data.read_text(encoding="utf-8", errors="replace").splitlines()
        header = [c.strip() for c in lines[0].split(",")] if lines else []
        missing = missing_quantity(user, header)
        if missing:
            return Outcome(None, infeasible=True, note=f"oneshot: question asks for {missing!r}, no such column")
        preview = "\n".join(lines[:PREVIEW_ROWS + 1])
        user += (f"\n\nFile {data.relative_to(workdir).as_posix()}: {max(len(lines) - 1, 0)} data rows; "
                 f"its only columns are: {', '.join(header)}. First lines:\n{preview}")
        system = DATA_PROMPT
    else:
        workdir = Path(tempfile.gettempdir()) / "harness_work" / task.task_id
        workdir.mkdir(parents=True, exist_ok=True)
        system = MATH_PROMPT

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(1, MAX_PROGRAM_ATTEMPTS + 1):
        reply = str(getattr(await model.invoke(messages), "content", "") or "")
        if says_infeasible(reply):
            return Outcome(None, infeasible=True, note="oneshot: model says infeasible", final_text=reply)
        code = extract_code(reply)
        if code is None:
            feedback = "Reply with one ```python block, or with the single word INFEASIBLE."
        else:
            exit_code, output = await _run_program(code, workdir, settings)
            if exit_code == 0:
                answer = normalize_answer(_last_line(output), task.check)
                if is_infeasible(answer):  # программа сама проверила данные и сообщила, что посчитать нельзя
                    return Outcome(None, infeasible=True, note="oneshot: program says infeasible", final_text=reply)
                if answer and (task.check != "exact_int" or re.fullmatch(r"-?\d+", answer)):
                    return Outcome(answer, local_ok=True, note=f"oneshot: program, {attempt} call(s)", final_text=reply)
                feedback = f"The program printed {answer!r}, but only the final answer is needed (an integer)."
            else:
                # Невыполнимость, которую видно по запуску, а не по словам модели.
                missing = _KEY_ERROR_RE.search(output)
                if task.task_type == "data_analysis" and missing and missing["key"] not in header:
                    return Outcome(None, infeasible=True, note=f"oneshot: no column {missing['key']!r}",
                                   final_text=reply)
                if task.task_type == "math" and "ZeroDivisionError" in output:
                    return Outcome(None, infeasible=True, note="oneshot: division by zero", final_text=reply)
                feedback = f"The program failed:\n```\n{output[-1500:]}\n```\nFix it. Use the same reply format."
        log.info("[%s] oneshot attempt %d failed: %s | reply: %r", task.task_id, attempt,
                 feedback.splitlines()[0], reply[:200])
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": feedback}]
    return None


async def _solve_search(task: Task, model) -> Outcome | None:
    kb = compact_kb(kb_files(task))
    user = f"{question_text(task.instruction)}\n\nKnowledge base (one entry per line):\n{kb}"
    reply = str(getattr(await model.invoke([{"role": "system", "content": SEARCH_PROMPT},
                                           {"role": "user", "content": user}]), "content", "") or "")
    found = _ANSWER_RE.findall(reply)
    if not found:
        return None
    answer = found[-1].strip().strip("`").strip()
    if is_infeasible(answer):
        return Outcome(None, infeasible=True, note="oneshot: model says infeasible", final_text=reply)
    if not answer or answer.casefold() not in kb.casefold():
        log.info("[%s] oneshot answer %r is not in the knowledge base", task.task_id, answer)
        return None
    return Outcome(answer, note="oneshot: knowledge base", final_text=reply)


async def solve_oneshot(task: Task, settings: Settings, meter: TokenMeter, model_name: str) -> Outcome | None:
    if task.task_type == "coding":
        return await solve_coding(task, settings, meter, model_name)
    if not supports_oneshot(task):
        return None
    from harness.llm import make_model  # openJiuwen нужен только здесь; помощники выше тестируются без него

    model = make_model(settings, meter, task.task_id, model_name)
    if task.task_type == "search":
        return await _solve_search(task, model)
    return await _solve_with_program(task, settings, model)
