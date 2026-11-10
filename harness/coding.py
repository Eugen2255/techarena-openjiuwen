"""coding за один вызов LLM: модель присылает исправленные функции, харнесс подставляет их и гоняет тесты.

Цикл ReAct для таких задач дорог: второй вызов пересылает весь контекст заново ради ответа DONE.
Здесь контекст сжат (исходник + выжимка упавших тестов), а ответ модели — только изменённые функции.

До вызова LLM бесплатно проверяем: тесты уже проходят → DONE; тесты противоречат друг другу → невыполнима.
Непривычная раскладка (несколько модулей, нет тестов) → None, задачу решает агент.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from harness.config import Settings
from harness.meter import TokenMeter
from harness.shell import run_command
from harness.tasks import Task

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
MAX_SOURCE_BYTES = 8000

SYSTEM_PROMPT = """You fix bugs in a Python module so that its unit tests pass.
Fix each function according to its name and intended meaning: hidden tests check general behaviour,
so never special-case the visible test values. Do not change the tests.
Reply ONLY with one ```python block containing the complete corrected definitions of the functions you changed.
If the tests contradict each other and cannot all pass, reply with the single word INFEASIBLE."""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_DROP_LINE_RE = re.compile(r"^(?:[=-]{10,}|Traceback \(most recent call last\):|  File \".*|[-+?] .*|\.*[FE.]*\.*)$")


@dataclass
class CodingOutcome:
    answer: str | None
    infeasible: bool = False
    local_ok: bool | None = None
    note: str = ""
    final_text: str = ""


def compact_test_output(output: str, limit: int = 1500) -> str:
    """Оставляет из вывода unittest имена упавших тестов, строки с assert и сообщения об ошибках."""
    lines = [line.rstrip() for line in output.splitlines()]
    kept = [line for line in lines if line.strip() and not _DROP_LINE_RE.match(line)]
    text = "\n".join(kept)
    return text if len(text) <= limit else text[:limit] + "\n..."


def contradictory_tests(test_sources: list[str]) -> bool:
    """Один и тот же вызов ожидается с разными литералами или и True, и False.

    Понимает assertEqual(f(x), 5), assertTrue(f(x) == 5) и пару assertTrue(e) / assertFalse(e).
    """
    expected: dict[str, set] = {}
    for source in test_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name, args = node.func.attr, node.args
            pair = None
            if name in ("assertEqual", "assertEquals") and len(args) >= 2:
                pair = args[0], args[1]
            elif (name == "assertTrue" and args and isinstance(args[0], ast.Compare)
                  and len(args[0].ops) == 1 and isinstance(args[0].ops[0], ast.Eq)):
                pair = args[0].left, args[0].comparators[0]
            if pair is not None:
                try:
                    value = ast.literal_eval(pair[1])
                except (ValueError, SyntaxError):
                    continue
                key = ast.dump(pair[0])
            elif name in ("assertTrue", "assertFalse") and args:
                key, value = ast.dump(args[0]), name == "assertTrue"
            else:
                continue
            expected.setdefault(key, set()).add(repr(value))
    return any(len(values) > 1 for values in expected.values())


def apply_functions(source: str, reply_code: str) -> str | None:
    """Заменяет в модуле функции (и классы) с теми же именами, новые дописывает в конец, импорты — в начало."""
    try:
        module, patch = ast.parse(source), ast.parse(reply_code)
    except SyntaxError:
        return None
    replacements, additions, imports = {}, [], []
    defined = {node.name: node for node in module.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in patch.body:
        segment = ast.get_source_segment(reply_code, node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in defined:
                replacements[node.name] = segment
            else:
                additions.append(segment)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if segment not in source:
                imports.append(segment)
    if not replacements and not additions:
        return None

    lines = source.splitlines()
    for name in sorted(replacements, key=lambda n: defined[n].lineno, reverse=True):
        node = defined[name]
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        lines[first:node.end_lineno] = replacements[name].splitlines()
    result = "\n".join(lines).rstrip() + "\n"
    if additions:
        result += "\n\n" + "\n\n\n".join(additions) + "\n"
    if imports:
        result = "\n".join(imports) + "\n" + result
    return result


async def _run_tests(tests: list[Path], workdir: Path, settings: Settings) -> tuple[bool, str]:
    outputs, ok = [], True
    for test in tests:
        code, output = await run_command(f"python3 {test.name}", workdir, settings.command_timeout_s, 4000)
        ok &= code == 0
        outputs.append(output)
    return ok, "\n".join(outputs)


def _extract_code(reply: str) -> str:
    blocks = _FENCE_RE.findall(reply)
    return "\n\n".join(blocks) if blocks else reply


async def solve_coding(task: Task, settings: Settings, meter: TokenMeter, model_name: str) -> CodingOutcome | None:
    workdir = task.workspace
    if workdir is None:
        return None
    tests = sorted(workdir.glob("test_*.py"))
    sources = [p for p in sorted(workdir.glob("*.py")) if not p.name.startswith("test_")]
    if not tests or len(sources) != 1 or sources[0].stat().st_size > MAX_SOURCE_BYTES:
        return None
    module = sources[0]
    test_texts = [t.read_text(encoding="utf-8", errors="replace") for t in tests]

    passed, output = await _run_tests(tests, workdir, settings)
    if passed:
        return CodingOutcome("DONE", local_ok=True, note="coding: tests already pass")
    if contradictory_tests(test_texts):
        return CodingOutcome(None, infeasible=True, note="coding: contradictory tests")

    from harness.llm import make_model  # openJiuwen нужен только здесь; помощники выше тестируются без него

    model = make_model(settings, meter, task.task_id, model_name)
    reply = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        source = module.read_text(encoding="utf-8", errors="replace")
        parts = [f"{module.name}:\n```python\n{source}```",
                 f"Failing tests (`python3 {tests[0].name}`):\n```\n{compact_test_output(output)}\n```"]
        if attempt > 1:  # первый раунд не помог — показываем тесты целиком
            parts.insert(1, "\n".join(f"{t.name}:\n```python\n{text}```" for t, text in zip(tests, test_texts)))
        message = await model.invoke([{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": "\n\n".join(parts)}])
        reply = str(getattr(message, "content", "") or "")
        if reply.strip().strip(".").upper() == "INFEASIBLE":
            return CodingOutcome(None, infeasible=True, note="coding: model says infeasible", final_text=reply)
        patched = apply_functions(source, _extract_code(reply))
        if patched is None:
            log.info("[%s] coding attempt %d: no usable code in reply", task.task_id, attempt)
            continue
        module.write_text(patched, encoding="utf-8")
        passed, output = await _run_tests(tests, workdir, settings)
        if passed:
            return CodingOutcome("DONE", local_ok=True, note=f"coding: fixed in {attempt} call(s)", final_text=reply)
    return CodingOutcome(None, local_ok=False, note="coding: tests still fail, answer withheld", final_text=reply)
