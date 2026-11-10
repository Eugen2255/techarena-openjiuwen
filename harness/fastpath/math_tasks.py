"""math: склад с цепочкой операций над запасом.

Шаблон из примеров:
  A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and it keeps
  one share (rounded down). Then the stock is doubled by merging with a sister warehouse. ...
  It ships 36 boxes per day for 3 days. It receives 2 deliveries of 87 boxes each.
  How many boxes does it have at the end? Compute it with the bash tool ...

Каждое предложение условия должно целиком совпасть с одним из шаблонов ниже, иначе — None (решит агент).
Если вместо числа стоит неопределённое количество («an unknown number of», «some»…), а всё остальное разобрано,
задача невыполнима: точно посчитать нельзя. Запас, ушедший в минус, и деление на 0 отдаём агенту.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from harness.fastpath import FastAnswer, register
from harness.tasks import Task

_WORDS = r"[a-z]+(?: [a-z]+)?"
# Число или неопределённое количество на его месте.
_VAGUE = r"an unknown number of|an unspecified number of|a number of|some|several|a few"
_QTY = rf"(?:\d+|{_VAGUE})"
_NUMBER_GROUPS = ("n", "parts", "count", "days", "size")
_QUESTION_RE = re.compile(rf"How many (?P<unit>{_WORDS}) does it have at the end\?")
_START_RE = re.compile(rf"An? {_WORDS} (?:starts with|has) (?P<n>{_QTY}) (?P<unit>{_WORDS})", re.IGNORECASE)
_CONNECTOR_RE = re.compile(r"^(?:then|next|after that|finally),?\s+", re.IGNORECASE)
_MULTIPLIERS = {"doubled": 2, "tripled": 3, "quadrupled": 4}

Step = Callable[[int], int | None]


def _split(m: re.Match) -> Step:
    parts, up = int(m["parts"]), m["rounding"].lower() == "up"
    if parts <= 0:
        return lambda x: None
    return lambda x: -(-x // parts) if up else x // parts


def _operations(unit: str) -> list[tuple[re.Pattern, Callable[[re.Match], Step]]]:
    """Шаблоны операций. Единица измерения должна совпадать с той, что в первом предложении."""
    u = re.escape(unit)
    q = _QTY
    patterns: list[tuple[str, Callable[[re.Match], Step]]] = [
        (rf"the stock is split equally into (?P<parts>{q}) \w+ and it keeps one share \(rounded (?P<rounding>down|up)\)",
         _split),
        (r"the stock is (?P<mult>doubled|tripled|quadrupled)(?: by merging with (?:a|two|three) sister warehouses?)?",
         lambda m: (lambda x, k=_MULTIPLIERS[m["mult"].lower()]: x * k)),
        (rf"it ships (?P<count>{q}) {u} per day for (?P<days>{q}) days?",
         lambda m: (lambda x, d=int(m["count"]) * int(m["days"]): x - d)),
        (rf"it receives (?P<count>{q}) deliver(?:y|ies) of (?P<size>{q}) {u} each",
         lambda m: (lambda x, d=int(m["count"]) * int(m["size"]): x + d)),
        (rf"it (?:ships|sells|loses) (?P<n>{q}) {u}",
         lambda m: (lambda x, d=int(m["n"]): x - d)),
        (rf"it (?:receives|buys) (?P<n>{q}) (?:more )?{u}",
         lambda m: (lambda x, d=int(m["n"]): x + d)),
    ]
    return [(re.compile(p, re.IGNORECASE), make) for p, make in patterns]


def _sentences(text: str) -> list[str]:
    return [s.strip().rstrip(".").strip() for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip()]


def _is_vague(match: re.Match) -> bool:
    """В каком-то числовом поле вместо числа неопределённое количество."""
    values = match.groupdict()
    return any(values.get(g) is not None and not values[g].isdigit() for g in _NUMBER_GROUPS)


@register("math")
def warehouse_stock(task: Task, workdir: Path) -> FastAnswer | None:
    question = _QUESTION_RE.search(task.instruction)
    if not question:
        return None
    # Условие — всё до вопроса; хвост инструкции («echo 42 > answer.txt» и т.п.) не трогаем.
    sentences = _sentences(task.instruction[:question.start()])
    if not sentences:
        return None
    start = _START_RE.fullmatch(sentences[0])
    if not start or start["unit"].lower() != question["unit"].lower():
        return None

    operations = _operations(start["unit"])
    steps: list[Step] = []
    vague = _is_vague(start)
    for sentence in sentences[1:]:
        body = _CONNECTOR_RE.sub("", sentence)
        for pattern, make in operations:
            match = pattern.fullmatch(body)
            if match:
                if _is_vague(match):
                    vague = True
                else:
                    steps.append(make(match))
                break
        else:
            return None  # незнакомая операция
    if vague:
        return FastAnswer(None, infeasible=True, note="warehouse_stock: unknown quantity")

    stock = int(start["n"])
    for step in steps:
        stock = step(stock)
        if stock is None or stock < 0:
            return None
    return FastAnswer(str(stock), note=f"warehouse_stock: {len(steps)} steps")
