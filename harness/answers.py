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


_WRAPPERS = (("**", "**"), ("__", "__"), ("*", "*"), ("`", "`"), ('"', '"'), ("'", "'"), ("«", "»"), ("“", "”"))
# Целое число в тексте: «1,200» или «42», «6668.0»; не часть десятичной дроби («3.5» числом не считаем).
_INT_TOKEN_RE = re.compile(r"(?<![\d.,])-?\d{1,3}(?:,\d{3})+(?![\d.])|(?<![\d.,])-?\d+(?:\.0+)?(?![\d.,])")


def normalize_answer(answer: str, check: str | None) -> str:
    """Ответ к виду, который пройдёт сравнение организаторов (strip + нижний регистр).

    Снимает обёртки модели (**Osaka**, `42`, "Osaka"), точку после ответа из одного слова («Osaka.»);
    exact_int: «6668.0» → «6668», «1,234» → «1234», «$1,200» и «42 boxes» → число, если оно в ответе одно.
    """
    answer = answer.strip()
    for _ in range(2):
        for left, right in _WRAPPERS:
            if len(answer) > len(left) + len(right) and answer.startswith(left) and answer.endswith(right):
                answer = answer[len(left):-len(right)].strip()
                break
    if re.fullmatch(r"[\w'-]+\.", answer):  # «Osaka.», «1983.»; «Acme Inc.» (несколько слов) не трогаем
        answer = answer[:-1]
    if check == "exact_int" and not re.fullmatch(r"-?\d+", answer):
        tokens = _INT_TOKEN_RE.findall(answer)
        if len(tokens) == 1:
            return tokens[0].replace(",", "").split(".")[0]
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
