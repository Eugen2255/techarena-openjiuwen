"""search: вопрос по базе знаний kb/*.md — человек → работодатель / наставник → компания.

Формат базы из примеров:
  person_*.md:  "# Irina Smith", "Born: 1983", "Employer: Dune Robotics", "Mentor: Alice Smith"
  company_*.md: "# Granite AI", "Headquarters: Lisbon", "Founded: 1982"
Вопросы из примеров:
  In which year was Irina Smith born?
  In which city is the headquarters of Sergei Garcia's employer?

Вопрос разбирается на шаблон атрибута и цепочку связей ("X's mentor's employer"). Ответ даём, только если он
однозначен. Невыполнимой задачу объявляем, только если вопрос разобран целиком, база ровно в ожидаемом формате,
а данных нет (человека нет в базе, ссылка ведёт в пустоту, нет поля, одинаковые имена дают разные ответы).
Во всех остальных случаях — None: решит агент.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.fastpath import FastAnswer, register
from harness.tasks import Task

RELATIONS = {"employer": "Employer", "mentor": "Mentor"}
KNOWN_FIELDS = {"Born", "Employer", "Mentor", "Headquarters", "Founded"}

_DIR_RE = re.compile(r"The directory (?P<dir>[\w./-]+?)/? contains")
_QUESTION_RE = re.compile(r"(?:^|(?<=[.?!] ))(?P<q>[A-Z][^.?!]*\?)")
_FULL_NAME_RE = re.compile(r"[A-Z][\w.-]*(?: [A-Z][\w.-]*)+")
_NAME_RE = re.compile(r"[A-Z][\w.-]*(?: [A-Z][\w.-]*)*")
_E = r"(?P<e>.+?)"

# Вопрос об атрибуте сущности: (шаблон, поле).
ATTRIBUTE_QUESTIONS = [(re.compile(p), f) for p, f in [
    (rf"In which year was {_E} born\?", "Born"),
    (rf"When was {_E} born\?", "Born"),
    (rf"What is the birth year of {_E}\?", "Born"),
    (rf"In which city is the headquarters of {_E}\?", "Headquarters"),
    (rf"Where is the headquarters of {_E}\?", "Headquarters"),
    (rf"(?:In which city|Where) is {_E} headquartered\?", "Headquarters"),
    (rf"In which year was {_E} founded\?", "Founded"),
    (rf"When was {_E} founded\?", "Founded"),
]]
# Вопрос, ответ на который — имя: (шаблон, связь, которую добавляет сам вопрос).
NAME_QUESTIONS = [(re.compile(p), r) for p, r in [
    (rf"Who is {_E}\?", None),  # "Who is X's mentor?" — связь уже в выражении
    (rf"(?:Which|What) company does {_E} work for\?", "employer"),
    (rf"Where does {_E} work\?", "employer"),
    (rf"(?:Which|What) company employs {_E}\?", "employer"),
    (rf"Who mentors {_E}\?", "mentor"),
]]


class Unresolvable(Exception):
    """Данных в базе не хватает для ответа."""


@dataclass
class KnowledgeBase:
    records: dict[str, list[dict[str, str]]] = field(default_factory=dict)  # casefold(название) -> записи
    clean: bool = True  # все файлы ровно в ожидаемом формате

    def lookup(self, name: str) -> list[dict[str, str]]:
        return self.records.get(name.casefold(), [])


def load_kb(kb_dir: Path) -> KnowledgeBase | None:
    files = sorted(kb_dir.rglob("*.md"))
    if not files:
        return None
    kb = KnowledgeBase()
    for path in files:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.strip()]
        if not lines or not lines[0].startswith("# "):
            kb.clean = False
            continue
        fields: dict[str, str] = {}
        for line in lines[1:]:
            match = re.fullmatch(r"([A-Za-z][A-Za-z ]*):\s*(.+)", line)
            if not match or match[1] not in KNOWN_FIELDS or match[1] in fields:
                kb.clean = False
            if match:
                fields.setdefault(match[1], match[2].strip())
        kb.records.setdefault(lines[0][2:].strip().casefold(), []).append(fields)
    return kb


def parse_entity(expr: str) -> tuple[str, list[str]] | None:
    """'Sergei Garcia's mentor's employer' -> ('Sergei Garcia', ['mentor', 'employer'])."""
    expr = expr.strip()
    match = re.fullmatch(r"the (employer|mentor) of (.+)", expr)
    if match:
        inner = parse_entity(match[2])
        return None if inner is None else (inner[0], inner[1] + [match[1]])
    base, *hops = re.split(r"['’]s ", expr)
    hops = [hop.strip().lower() for hop in hops]
    if not _NAME_RE.fullmatch(base) or any(hop not in RELATIONS for hop in hops):
        return None
    return base, hops


def resolve(kb: KnowledgeBase, base: str, hops: list[str], final_field: str | None) -> set[str]:
    """Все возможные ответы (их больше одного, если в базе одинаковые имена)."""
    current = {base}
    for hop in hops:
        following = set()
        for name in current:
            records = kb.lookup(name)
            if not records:
                raise Unresolvable(f"no entry for {name!r}")
            for record in records:
                if RELATIONS[hop] not in record:
                    raise Unresolvable(f"{name!r} has no {hop}")
                following.add(record[RELATIONS[hop]])
        current = following
    if final_field is None:
        return current
    values = set()
    for name in current:
        records = kb.lookup(name)
        if not records:
            raise Unresolvable(f"no entry for {name!r}")
        for record in records:
            if final_field not in record:
                raise Unresolvable(f"{name!r} has no {final_field}")
            values.add(record[final_field])
    return values


def parse_question(question: str) -> tuple[str, list[str], str | None] | None:
    """Вопрос -> (имя, цепочка связей, поле ответа или None для ответа-имени)."""
    for pattern, final_field in ATTRIBUTE_QUESTIONS:
        match = pattern.fullmatch(question)
        if match and (entity := parse_entity(match["e"])):
            return entity[0], entity[1], final_field
    for pattern, relation in NAME_QUESTIONS:
        match = pattern.fullmatch(question)
        if match and (entity := parse_entity(match["e"])):
            hops = entity[1] + ([relation] if relation else [])
            if hops:  # «Who is Irina Smith?» без связи — не наш шаблон
                return entity[0], hops, None
    return None


@register("search")
def knowledge_base_lookup(task: Task, workdir: Path) -> FastAnswer | None:
    dir_match = _DIR_RE.search(task.instruction)
    question_match = _QUESTION_RE.search(task.instruction)
    if not dir_match or not question_match:
        return None
    parsed = parse_question(question_match["q"])
    kb = load_kb(workdir / dir_match["dir"])
    if parsed is None or kb is None:
        return None
    base, hops, final_field = parsed

    try:
        values = resolve(kb, base, hops, final_field)
    except Unresolvable as exc:
        # Невыполнимой считаем только при базе в ожидаемом формате и полном имени (а не «Irina's employer»).
        if kb.clean and _FULL_NAME_RE.fullmatch(base):
            return FastAnswer(None, infeasible=True, note=f"knowledge_base_lookup: {exc}")
        return None
    if len(values) == 1:
        return FastAnswer(values.pop(), note=f"knowledge_base_lookup: {len(hops)} hops")
    if kb.clean:
        return FastAnswer(None, infeasible=True, note=f"knowledge_base_lookup: ambiguous {sorted(values)}")
    return None
