"""Учёт токенов по правилам соревнования: стоимость = выходные + входные / 10."""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

INPUT_WEIGHT = 0.1  # 10 входных токенов = 1 выходной


class BudgetExceeded(RuntimeError):
    """Новый вызов LLM запрещён: исчерпан общий бюджет или потолок на задачу."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def cost(self) -> float:
        # completion_tokens у NeuralDeep уже включает рассуждения (проверено 10.11: qwen3.8-27b вернула
        # completion_tokens=62 при reasoning_tokens=56), поэтому reasoning отдельно не прибавляем.
        return self.output_tokens + self.input_tokens * INPUT_WEIGHT

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": round(self.cost, 1),
        }


class TokenMeter:
    """Общий счётчик на весь прогон. Работает в одном event loop, блокировки не нужны.

    Остатка прокси не сообщает: расход считают организаторы и при исчерпании бюджета останавливают прогон
    (встреча 09.11), поэтому пересекать бюджет нельзя ни на один вызов. Перед каждым вызовом резервируем его
    худшую стоимость (reserve), после вызова снимаем резерв (release) и записываем фактический расход (record),
    а для вызова, оборвавшегося ошибкой, — оценку сверху (charge).
    """

    def __init__(self, budget: float | None = None, task_cap: float | None = None):
        self.budget = budget
        self.task_cap = task_cap
        self.total = Usage()
        self._per_task: dict[str, Usage] = {}
        self._reserved = 0.0

    def record(self, task_id: str, usage_metadata) -> None:
        if usage_metadata is None:
            log.warning("[%s] LLM response without usage, cost is not counted", task_id)
            return
        usage = Usage(
            calls=1,
            input_tokens=int(getattr(usage_metadata, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage_metadata, "output_tokens", 0) or 0),
            reasoning_tokens=int(getattr(usage_metadata, "reasoning_tokens", 0) or 0),
        )
        self.total.add(usage)
        self._per_task.setdefault(task_id, Usage()).add(usage)

    def charge(self, task_id: str, input_tokens: float, output_tokens: float = 0) -> None:
        """Вызов, оборвавшийся ошибкой, — без usage. Ушедший запрос прокси организаторов всё равно считает
        (встреча 09.11), поэтому списываем оценку сверху."""
        usage = Usage(calls=1, input_tokens=round(input_tokens), output_tokens=round(output_tokens))
        self.total.add(usage)
        self._per_task.setdefault(task_id, Usage()).add(usage)

    def task_usage(self, task_id: str) -> Usage:
        return self._per_task.get(task_id, Usage())

    def remaining(self) -> float | None:
        """Сколько ещё можно потратить по факту (без учёта резервов идущих вызовов)."""
        return None if self.budget is None else self.budget - self.total.cost

    def exhausted(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def fits(self, task_id: str, worst_case: float) -> bool:
        """Поместится ли вызов, когда закончатся идущие вызовы (то есть без учёта чужих резервов)."""
        if self.task_cap is not None and self.task_usage(task_id).cost >= self.task_cap:
            return False
        remaining = self.remaining()
        return remaining is None or worst_case <= remaining

    def reserve(self, task_id: str, worst_case: float) -> None:
        """Перед запросом к LLM: резервирует худшую стоимость вызова или запрещает вызов."""
        spent = self.task_usage(task_id).cost
        if self.task_cap is not None and spent >= self.task_cap:
            raise BudgetExceeded(f"task cap {self.task_cap:.0f} reached ({spent:.0f} spent)")
        remaining = self.remaining()
        if remaining is not None and worst_case > remaining - self._reserved:
            raise BudgetExceeded(f"call may cost up to {worst_case:.0f}, only {remaining - self._reserved:.0f} "
                                 f"of {self.budget:.0f} left (incl. calls in progress)")
        self._reserved += worst_case

    def release(self, worst_case: float) -> None:
        self._reserved = max(0.0, self._reserved - worst_case)
