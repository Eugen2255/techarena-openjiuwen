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
import textwrap
from pathlib import Path

from harness.answers import Outcome, says_infeasible
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
Reply ONLY with one ```python block containing the complete corrected definitions of the functions you changed
(if you change a method, send its whole class).
If the tests contradict each other and cannot all pass, reply with the single word INFEASIBLE."""

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_DROP_LINE_RE = re.compile(r"^(?:[=-]{10,}|Traceback \(most recent call last\):|  File \".*|[-+?] .*|\.*[FE.]*\.*)$")


def compact_test_output(output: str, limit: int = 1500) -> str:
    """Оставляет из вывода unittest имена упавших тестов, строки с assert и сообщения об ошибках."""
    lines = [line.rstrip() for line in output.splitlines()]
    kept = [line for line in lines if line.strip() and not _DROP_LINE_RE.match(line)]
    text = "\n".join(kept)
    return text if len(text) <= limit else text[:limit] + "\n..."


# Функции, у которых два одинаковых вызова законно дают разное (тест «значения различаются» не противоречив).
_NONDETERMINISTIC_RE = re.compile(r"rand|uuid|time|token|nonce|shuffle|choice|sample|unique|counter|next_id",
                                  re.IGNORECASE)
# Импорты тестов, которые не являются проверяемым модулем.
_NOT_TESTED = {"unittest", "pytest", "sys", "os", "math", "random", "re", "io", "json", "collections", "itertools",
               "functools", "typing", "pathlib", "string", "decimal", "fractions", "copy", "importlib"}


def _tested_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Под какими именами тесты видят проверяемый модуль: (алиасы модуля, функции из from-импорта)."""
    modules: set[str] = set()
    functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name.split(".")[0] for alias in node.names
                           if alias.name.split(".")[0] not in _NOT_TESTED)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] not in _NOT_TESTED:
            functions.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return modules, functions


def stateful_functions(module_source: str) -> set[str]:
    """Функции модуля, которые на одинаковый вызов могут вернуть разное: global/nonlocal, изменяемые значения
    по умолчанию, обращение к переменным уровня модуля (кэш, счётчик)."""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return set()
    module_vars = {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                   for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                   if isinstance(target, ast.Name)}
    result = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
        if (any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node))
                or any(not isinstance(d, ast.Constant) for d in defaults)
                or any(isinstance(n, ast.Name) and n.id in module_vars for n in ast.walk(node))):
            result.add(node.name)
    return result


def _tested_call(expr: ast.expr, modules: set[str], functions: set[str], stateful: set[str]) -> bool:
    """Вызов функции проверяемого модуля с литералами в аргументах: m.f(1) или f(1) после from-импорта.

    Переменные (result), методы объектов (s.pop()), классы, вызовы без аргументов и функции с состоянием
    не берём: там одинаковый текст законно даёт разные значения."""
    if not isinstance(expr, ast.Call) or not (expr.args or expr.keywords):
        return False
    func = expr.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in modules:
        name = func.attr
    elif isinstance(func, ast.Name) and func.id in functions:
        name = func.id
    else:
        return False
    if name[:1].isupper() or name in stateful or _NONDETERMINISTIC_RE.search(name):
        return False
    try:
        for arg in [*expr.args, *(k.value for k in expr.keywords)]:
            ast.literal_eval(arg)
    except (ValueError, SyntaxError, TypeError):
        return False
    return True


def _comparison(name: str, args: list[ast.expr]) -> tuple[ast.expr, ast.expr, bool] | None:
    """(левая часть, правая часть, ожидается равенство?) для assertEqual/NotEqual и assertTrue/False(a ==/!= b)."""
    if name in ("assertEqual", "assertEquals") and len(args) >= 2:
        return args[0], args[1], True
    if name in ("assertNotEqual", "assertNotEquals") and len(args) >= 2:
        return args[0], args[1], False
    if name in ("assertTrue", "assertFalse") and args and isinstance(args[0], ast.Compare) \
            and len(args[0].ops) == 1 and isinstance(args[0].ops[0], (ast.Eq, ast.NotEq)):
        equal = isinstance(args[0].ops[0], ast.Eq) == (name == "assertTrue")
        return args[0].left, args[0].comparators[0], equal
    return None


def contradictory_tests(test_sources: list[str], module_source: str | None = None) -> bool:
    """Тесты не может пройти никакая честная реализация.

    Смотрим только на вызовы функций проверяемого модуля с литералами в аргументах (см. _tested_call):
    один вызов ожидается с разными значениями (assertEqual(f(x), 5) и assertTrue(f(x) == 6)); значение и
    ожидается, и запрещено; пара assertTrue(e) / assertFalse(e); самосравнение (assertNotEqual(f(1), f(1))).
    Правило нарочно осторожное: ложное «противоречие» стоит баллов за задачу, а верное экономит один вызов LLM
    (баллов за распознанную невыполнимую задачу нет). module_source — исходник модуля, чтобы не трогать
    функции с состоянием.
    """
    stateful = stateful_functions(module_source) if module_source else set()
    expected: dict[str, list] = {}
    forbidden: dict[str, list] = {}
    truth: dict[str, set] = {}
    for source in test_sources:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        modules, functions = _tested_names(tree)

        def tested(expr: ast.expr) -> bool:
            return _tested_call(expr, modules, functions, stateful)

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            name, args = node.func.attr, node.args
            comparison = _comparison(name, args)
            if comparison is not None:
                left, right, equal = comparison
                if not tested(left) and tested(right):
                    left, right = right, left
                if not tested(left):
                    continue
                if not equal and ast.dump(left) == ast.dump(right):
                    return True
                try:
                    value = ast.literal_eval(right)
                except (ValueError, SyntaxError, TypeError):
                    continue
                (expected if equal else forbidden).setdefault(ast.dump(left), []).append(value)
            elif name in ("assertTrue", "assertFalse") and args and tested(args[0]):
                truth.setdefault(ast.dump(args[0]), set()).add(name == "assertTrue")
    # Сравниваем по ==, а не по тексту: 1 и 1.0 — одно и то же значение.
    return (any(a != b for values in expected.values() for a in values for b in values)
            or any(e == f for key, values in forbidden.items() for f in values for e in expected.get(key, []))
            or any(len(v) > 1 for v in truth.values()))


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFS = (*_FUNCS, ast.ClassDef)


def _span(node: ast.AST) -> tuple[int, int]:
    """Строки определения вместе с декораторами — как срез lines[start:end]."""
    return min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1, node.end_lineno


def _block(code: str, node: ast.AST, indent: int) -> list[str]:
    """Определение из ответа модели (с декораторами), сдвинутое на indent пробелов — например, внутрь класса."""
    start, end = _span(node)
    text = textwrap.dedent("\n".join(code.splitlines()[start:end]))
    return textwrap.indent(text, " " * indent).splitlines()


def apply_functions(source: str, reply_code: str) -> str | None:
    """Заменяет в модуле функции и классы с теми же именами, новые дописывает в конец, импорты — в начало.

    Метод, присланный без класса (def pop(self): ...) или в неполном классе, ставится на место внутри своего класса.
    """
    try:
        module, patch = ast.parse(source), ast.parse(reply_code)
    except SyntaxError:
        return None
    defined = {node.name: node for node in module.body if isinstance(node, _DEFS)}
    methods: dict[str, list[ast.AST]] = {}
    for cls in module.body:
        if isinstance(cls, ast.ClassDef):
            for item in cls.body:
                if isinstance(item, _FUNCS):
                    methods.setdefault(item.name, []).append(item)
    edits: list[tuple[int, int, list[str]]] = []  # (начало, конец, новые строки) — замена среза lines
    additions, imports = [], []
    for node in patch.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(reply_code, node)
            if segment not in source:
                imports.append(segment)
            continue
        if not isinstance(node, _DEFS):
            continue
        old = defined.get(node.name)
        if isinstance(node, ast.ClassDef) and isinstance(old, ast.ClassDef):
            old_methods = {i.name: i for i in old.body if isinstance(i, _FUNCS)}
            new_methods = [i for i in node.body if isinstance(i, _FUNCS)]
            if not new_methods or set(old_methods) <= {m.name for m in new_methods}:
                edits.append((*_span(old), _block(reply_code, node, old.col_offset)))  # класс целиком
                continue
            for method in new_methods:  # неполный класс: только присланные методы
                target = old_methods.get(method.name)
                if target is not None:
                    edits.append((*_span(target), _block(reply_code, method, target.col_offset)))
                else:
                    indent = next(iter(old_methods.values())).col_offset if old_methods else old.col_offset + 4
                    edits.append((old.end_lineno, old.end_lineno, ["", *_block(reply_code, method, indent)]))
        elif old is not None:
            edits.append((*_span(old), _block(reply_code, node, old.col_offset)))
        elif (isinstance(node, _FUNCS) and len(methods.get(node.name, [])) == 1
              and node.args.args and node.args.args[0].arg == "self"):
            target = methods[node.name][0]  # метод прислан без класса
            edits.append((*_span(target), _block(reply_code, node, target.col_offset)))
        else:
            additions.append("\n".join(_block(reply_code, node, 0)))
    if not edits and not additions:
        return None

    lines = source.splitlines()
    for start, end, new_lines in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start:end] = new_lines
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


async def solve_coding(task: Task, settings: Settings, meter: TokenMeter, model_name: str) -> Outcome | None:
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
        return Outcome("DONE", local_ok=True, note="coding: tests already pass")
    if contradictory_tests(test_texts, module.read_text(encoding="utf-8", errors="replace")):
        return Outcome(None, infeasible=True, note="coding: contradictory tests")

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
        if says_infeasible(reply):
            return Outcome(None, infeasible=True, note="coding: model says infeasible", final_text=reply)
        patched = apply_functions(source, _extract_code(reply))
        if patched is None:
            log.info("[%s] coding attempt %d: no usable code in reply", task.task_id, attempt)
            continue
        module.write_text(patched, encoding="utf-8")
        passed, output = await _run_tests(tests, workdir, settings)
        if passed:
            return Outcome("DONE", local_ok=True, note=f"coding: fixed in {attempt} call(s)", final_text=reply)
    return Outcome(None, local_ok=False, note="coding: tests still fail, answer withheld", final_text=reply)
