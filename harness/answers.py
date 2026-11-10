"""Приведение ответа к формату проверки. Без внешних зависимостей."""
from __future__ import annotations

import re


def normalize_answer(answer: str, check: str | None) -> str:
    """exact_int: «6668.0» → «6668», «1,234» → «1234» — модель иногда печатает число не в том виде."""
    answer = answer.strip()
    if check == "exact_int":
        if re.fullmatch(r"-?\d+\.0+", answer):
            return answer.split(".")[0]
        if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", answer):
            return answer.replace(",", "")
    return answer
