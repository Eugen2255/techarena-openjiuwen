"""Ответы: итог решения, приведение к формату проверки, признаки невыполнимости. Без внешних зависимостей."""
from __future__ import annotations

import re
from dataclasses import dataclass

INFEASIBLE = "INFEASIBLE"
_INFEASIBLE_MARKERS = {INFEASIBLE, "IMPOSSIBLE", "UNSOLVABLE", "CANNOT BE DETERMINED", "UNKNOWN", "N/A"}
# С этих фраз в инструкции начинается служебный хвост (как работать и куда писать ответ) — в запрос к LLM он не нужен.
_INSTRUCTION_TAIL_MARKERS = ("Use bash", "Use the file tools", "Compute it with", "Work only inside",
                             "When you are done")


@dataclass
class Outcome:
    """Итог решения одним вызовом (coding, oneshot). answer=None — answer.txt не создаём."""
    answer: str | None
    infeasible: bool = False
    local_ok: bool | None = None
    note: str = ""
    final_text: str = ""


def normalize_answer(answer: str, check: str | None) -> str:
    """exact_int: «6668.0» → «6668», «1,234» → «1234» — модель иногда печатает число не в том виде."""
    answer = answer.strip()
    if check == "exact_int":
        if re.fullmatch(r"-?\d+\.0+", answer):
            return answer.split(".")[0]
        if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", answer):
            return answer.replace(",", "")
    return answer


def is_infeasible(answer: str) -> bool:
    return answer.strip().strip(".").strip("`").upper() in _INFEASIBLE_MARKERS


def says_infeasible(reply: str) -> bool:
    """Ответ модели на запрос «код или INFEASIBLE»: без блока кода, но со словом INFEASIBLE (часто с пояснением)."""
    return is_infeasible(reply) or ("```" not in reply and re.search(rf"\b{INFEASIBLE}\b", reply) is not None)


def question_text(instruction: str) -> str:
    """Условие задачи без служебного хвоста инструкции."""
    cut = min((i for i in (instruction.find(m) for m in _INSTRUCTION_TAIL_MARKERS) if i > 0), default=len(instruction))
    return instruction[:cut].strip()
