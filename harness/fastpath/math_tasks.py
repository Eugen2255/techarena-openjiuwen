"""math: склад с цепочкой операций над запасом.

Шаблон из примеров:
  A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and it keeps
  one share (rounded down). Then the stock is doubled by merging with a sister warehouse. ...
  It ships 36 boxes per day for 3 days. It receives 2 deliveries of 87 boxes each.
  How many boxes does it have at the end? Compute it with the bash tool ...

Каждое предложение условия должно целиком совпасть с одним из шаблонов ниже, иначе — None (решит агент).
Задача невыполнима (точно посчитать нельзя), если всё условие разобрано, но где-то вместо числа неопределённое
количество («an unknown number of», «some»…), запас делят на 0 складов, или операции не хватает числа: темп без
длительности («ships 40 boxes per day»), деление без числа складов («split equally among its sister warehouses»).
Модель в таких случаях молча додумывает (1 день, 2 склада). Запас, ушедший в минус, отдаём агенту.
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


def _incomplete_operations(unit: str) -> list[tuple[re.Pattern, re.Pattern]]:
    """Операции, которым не хватает числа: (шаблон операции, шаблон предложения, которое могло бы это число дать).

    Если такое предложение среди незнакомых есть, невыполнимость не объявляем: может, число там.
    """
    u = re.escape(unit)
    return [(re.compile(op, re.IGNORECASE), re.compile(supplier, re.IGNORECASE)) for op, supplier in (
        (rf"it (?:ships|sells|loses|receives|gets|buys) (?:{_QTY}) {u} (?:per|every|each|a) day",
         r"\b(?:days?|weeks?|months?|years?|fortnight)\b"),
        (r"the stock is split equally (?:among|between|across) (?:all |its |the )?(?:sister |other )?\w+"
         r" and it keeps one share(?: \(rounded (?:down|up)\))?",
         r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten)\b.*\b(?:warehouses|parts|shares|stores|sisters)\b"),
    )]


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
    incomplete_ops = _incomplete_operations(start["unit"])
    steps: list[Step] = []
    vague = _is_vague(start)
    zero_split = False
    suppliers: list[re.Pattern] = []  # что должно найтись в незнакомых предложениях, чтобы число всё же было
    unknown: list[str] = []
    for sentence in sentences[1:]:
        body = _CONNECTOR_RE.sub("", sentence)
        for pattern, make in operations:
            match = pattern.fullmatch(body)
            if match:
                if _is_vague(match):
                    vague = True
                elif match.groupdict().get("parts") == "0":
                    zero_split = True  # деление на 0 складов не определено
                else:
                    steps.append(make(match))
                break
        else:
            gap = next((supplier for op, supplier in incomplete_ops if op.fullmatch(body)), None)
            if gap is not None:
                suppliers.append(gap)
            else:
                unknown.append(body)
    # Операции не хватает числа, и ни одно незнакомое предложение не может его дать — посчитать нельзя.
    if suppliers and not any(s.search(text) for s in suppliers for text in unknown):
        return FastAnswer(None, infeasible=True, note="warehouse_stock: operation lacks a number")
    if unknown:
        return None  # незнакомые операции — решит модель
    if vague:
        return FastAnswer(None, infeasible=True, note="warehouse_stock: unknown quantity")
    if zero_split:
        return FastAnswer(None, infeasible=True, note="warehouse_stock: split into 0 warehouses")

    stock = int(start["n"])
    for step in steps:
        stock = step(stock)
        if stock is None or stock < 0:
            return None
    return FastAnswer(str(stock), note=f"warehouse_stock: {len(steps)} steps")
