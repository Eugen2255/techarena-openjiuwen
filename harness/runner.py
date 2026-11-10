"""Главный цикл: быстрый путь без LLM, затем агент; параллельные воркеры, сторож времени и бюджета."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

from harness import estimator, fastpath
from harness.config import Settings, parse_settings
from harness.estimator import Estimate
from harness.meter import TokenMeter
from harness.solver import SolveResult, solve, workdir_for
from harness.tasks import Task, load_tasks
from harness.writer import AnswerWriter

log = logging.getLogger("harness")

# Столько ошибок LLM подряд (без единого ответа) считаем признаком, что прокси больше не отвечает.
MAX_LLM_ERRORS_IN_ROW = 5


def plan(tasks: list[Task], estimates: dict[str, Estimate]) -> list[Task]:
    """Порядок решения агентом: больше баллов на токен — раньше. TODO: учесть P(успех)."""
    return sorted(tasks, key=lambda t: t.reward / max(estimates[t.task_id].cost, 1.0), reverse=True)


class TrajectoryLog:
    """Траектории в jsonl: по строке на задачу. Нужны для оценщика и README."""

    def __init__(self, log_dir: Path):
        self.path = None
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.path = log_dir / f"trajectories_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        except OSError as exc:
            log.warning("trajectories are disabled: %s", exc)

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Run:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.meter = TokenMeter(settings.token_budget, settings.task_token_cap)
        self.writer = AnswerWriter(settings.out_file)
        self.trajectories = TrajectoryLog(settings.log_dir)
        self.stats = {"answered": 0, "fastpath": 0, "infeasible": 0, "skipped_budget": 0, "llm_errors_in_row": 0}

    def finish(self, task: Task, result: SolveResult, estimate: Estimate | None = None) -> None:
        if result.answer is not None:
            self.writer.save(task, result.answer)
            self.stats["answered"] += 1
        self.stats["infeasible"] += result.infeasible
        record = result.as_record()
        # Прогноз до запуска рядом с фактом — для проверки и дообучения оценщика.
        record["route"] = estimate.route if estimate else "fastpath"
        record["estimate"] = round(estimate.cost) if estimate else 0
        record["estimate_upper"] = round(estimate.upper) if estimate else 0
        self.trajectories.write(record)
        log.info("[%s] %s answer=%r%s cost=%.0f %.1fs%s", task.task_id, result.solver, result.answer,
                 " INFEASIBLE" if result.infeasible else "", result.usage.cost, result.seconds,
                 f" error={result.error}" if result.error else "")

    def run_fastpath(self, tasks: list[Task]) -> list[Task]:
        """Решает без LLM всё, что умеет. Возвращает задачи, оставшиеся агенту."""
        rest = []
        for task in tasks:
            started = time.monotonic()
            fast = fastpath.try_solve(task, workdir_for(task))
            if fast is None:
                rest.append(task)
                continue
            self.stats["fastpath"] += 1
            self.finish(task, SolveResult(task.task_id, task.task_type, task.reward, solver="fastpath",
                                          answer=fast.answer, infeasible=fast.infeasible, note=fast.note,
                                          seconds=time.monotonic() - started))
        return rest

    async def run_llm(self, tasks: list[Task]) -> int:
        """Решает задачи агентом в нескольких воркерах. Возвращает число неначатых задач."""
        settings, meter = self.settings, self.meter
        params = estimator.load_params()
        estimates = {t.task_id: estimator.estimate(t, params) for t in tasks}
        log.info("estimated cost of the remaining %d tasks: %.0f tokens (upper %.0f)", len(tasks),
                 sum(e.cost for e in estimates.values()), sum(e.upper for e in estimates.values()))
        queue = deque(plan(tasks, estimates))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.time_limit_s
        stop = asyncio.Event()

        async def worker() -> None:
            while queue and not stop.is_set():
                left = deadline - loop.time()
                if left < 5 or meter.exhausted():
                    stop.set()
                    break
                task = queue.popleft()
                estimate = estimates[task.task_id]
                remaining = meter.remaining()
                if remaining is not None and remaining < estimate.cost:
                    self.stats["skipped_budget"] += 1
                    log.info("[%s] skipped: estimate %.0f > remaining budget %.0f",
                             task.task_id, estimate.cost, remaining)
                    continue
                try:
                    result = await asyncio.wait_for(solve(task, settings, meter),
                                                    timeout=min(settings.task_timeout_s, left))
                except asyncio.TimeoutError:
                    result = SolveResult(task.task_id, task.task_type, task.reward, solver="llm",
                                         model=settings.model, error="timeout",
                                         usage=meter.task_usage(task.task_id))
                self.finish(task, result, estimate)

                if result.error and result.usage.calls == 0:
                    self.stats["llm_errors_in_row"] += 1
                    if self.stats["llm_errors_in_row"] >= MAX_LLM_ERRORS_IN_ROW:
                        log.error("LLM proxy stopped answering, stopping the run")
                        stop.set()
                else:
                    self.stats["llm_errors_in_row"] = 0

        await asyncio.gather(*(worker() for _ in range(max(1, settings.workers))))
        return len(queue)


async def run(settings: Settings) -> dict:
    tasks = load_tasks(settings.tasks_dir)
    log.info("loaded %d tasks from %s, budget %.0f tokens", len(tasks), settings.tasks_dir, settings.token_budget)
    state = Run(settings)
    rest = state.run_fastpath(tasks)
    log.info("fastpath solved %d tasks, %d left for the agent", len(tasks) - len(rest), len(rest))
    not_started = await state.run_llm(rest)
    stats = state.stats
    return {
        "tasks": len(tasks),
        "answered": stats["answered"],
        "fastpath": stats["fastpath"],
        "infeasible": stats["infeasible"],
        "skipped_budget": stats["skipped_budget"],
        "not_started": not_started,
        "budget": settings.token_budget,
        "usage": state.meter.total.as_dict(),
        "trajectories": str(state.trajectories.path) if state.trajectories.path else None,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("HARNESS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Построчные логи HTTP-клиента и openJiuwen на каждый вызов не нужны.
    for noisy in ("httpx", "openjiuwen"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("llm").setLevel(logging.CRITICAL)  # печатает весь промпт при каждой ошибке вызова
    settings = parse_settings(argv)
    if not settings.llm_base_url or not settings.llm_api_key:
        log.error("LLM_BASE_URL and LLM_API_KEY must be set")
        return 2
    if not settings.tasks_dir.is_dir():
        log.error("tasks dir not found: %s", settings.tasks_dir)
        return 2
    summary = asyncio.run(run(settings))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0
