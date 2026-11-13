"""Настройки харнесса: аргументы командной строки и переменные окружения.

На официальном прогоне передаются только LLM_BASE_URL и LLM_API_KEY,
поэтому значения по умолчанию для HARNESS_* должны быть боевыми.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Модели, доступные через прокси организаторов (docs/regulations.md).
MODELS = (
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
)
DEFAULT_MODEL = "qwen3.8-27b-noreason"
# math — модель с рассуждениями: на задачах нового вида 68/69 против 58/69 у -noreason при ×3 токенах
# (путает отданную и оставшуюся часть: «half are given away», «one third is moved»). README, «Выбор модели».
MATH_MODEL = "qwen3.8-27b"
DEFAULT_BUDGET = 100_000  # бюджет соревнования; в команде запуска BUDGET сейчас не передаётся


def is_reasoning(model: str) -> bool:
    """Модель с рассуждениями: её токены рассуждений входят в выход, и ей нужен больший потолок выхода."""
    return not model.endswith("-noreason")


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path
    out_file: Path | None
    llm_base_url: str
    llm_api_key: str
    verify_ssl: bool
    model: str
    math_model: str
    workers: int
    time_limit_s: float
    task_timeout_s: float
    command_timeout_s: float
    max_steps: int
    max_tool_output: int
    token_budget: float
    task_token_cap: float
    requests_per_minute: float
    llm_concurrency: int
    max_output_tokens: int
    max_output_tokens_reasoning: int
    log_dir: Path


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


_BUDGET_RE = re.compile(r"(\d+(?:\.\d+)?(?:e[+-]?\d+)?)([km]?)(?:tokens?)?")


def parse_budget(value: str | None) -> float | None:
    """Бюджет в том виде, в каком его могут передать в BUDGET: «100000», «1e5», «100_000», «100 000», «100,000»,
    «100k», «0.1M». Нечитаемое или неположительное значение — None: лучше бюджет по умолчанию, чем падение при старте."""
    text = (value or "").strip().lower().replace(" ", "").replace("_", "").replace(",", "")
    match = _BUDGET_RE.fullmatch(text)
    if not match:
        if text:
            logging.getLogger(__name__).warning("cannot parse budget %r, using the default", value)
        return None
    budget = float(match[1]) * {"": 1, "k": 1e3, "m": 1e6}[match[2]]
    return budget if budget > 0 else None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def parse_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description="TechArena harness on openJiuwen")
    parser.add_argument("--tasks-dir", default=os.getenv("TASKS", "/data/tasks"),
                        help="каталог с задачами (по умолчанию $TASKS или /data/tasks)")
    parser.add_argument("--out", default=os.getenv("OUT") or None,
                        help="файл с решениями, перезаписывается после каждой задачи (по умолчанию $OUT)")
    args = parser.parse_args(argv)

    return Settings(
        tasks_dir=Path(args.tasks_dir),
        out_file=Path(args.out) if args.out else None,
        llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        verify_ssl=os.getenv("HARNESS_VERIFY_SSL", "1") != "0",
        model=os.getenv("HARNESS_MODEL", DEFAULT_MODEL),
        math_model=os.getenv("HARNESS_MATH_MODEL", MATH_MODEL),
        workers=_env_int("HARNESS_WORKERS", 5),
        # Прогон обрывают через час; оставляем запас на завершение.
        time_limit_s=_env_float("HARNESS_TIME_LIMIT", 55 * 60),
        task_timeout_s=_env_float("HARNESS_TASK_TIMEOUT", 300),
        command_timeout_s=_env_float("HARNESS_COMMAND_TIMEOUT", 60),
        max_steps=_env_int("HARNESS_MAX_STEPS", 6),
        max_tool_output=_env_int("HARNESS_MAX_TOOL_OUTPUT", 2000),
        # Бюджет берём из BUDGET: организаторы предупредили, что он может отличаться от 100K. Остаток прокси не сообщает.
        token_budget=(parse_budget(os.getenv("BUDGET")) or parse_budget(os.getenv("HARNESS_TOKEN_BUDGET"))
                      or DEFAULT_BUDGET),
        # Потолок на одну задачу: дальше задача считается слишком дорогой и бросается.
        task_token_cap=_env_float("HARNESS_TASK_TOKEN_CAP", 6000),
        # Прокси — NeuralDeep, лимит 60 запросов в минуту; держимся ниже, чтобы не ловить 429.
        requests_per_minute=_env_float("HARNESS_RPM", 50),
        # NeuralDeep пускает не больше 3 одновременных запросов на аккаунт (проверено 10.11, ответ 429).
        llm_concurrency=_env_int("HARNESS_LLM_CONCURRENCY", 3),
        # Потолок выхода одного вызова: ограничивает худшую стоимость вызова для резерва бюджета.
        # Для моделей с рассуждениями его придётся поднять: рассуждения входят в выходные токены.
        # Потолок выхода резервируется до вызова: чем он ниже, тем меньше «мёртвый хвост» бюджета в конце прогона.
        # Замеры 13.11 (-noreason): медиана 73 токена на вызов, максимум 167.
        max_output_tokens=_env_int("HARNESS_MAX_OUTPUT_TOKENS", 1024),
        # Рассуждения входят в выход: на math медиана 215 токенов, максимум 554; потолок с большим запасом —
        # модель с рассуждениями работает, только когда бюджет не узкое место (runner.adapt_to_budget).
        max_output_tokens_reasoning=_env_int("HARNESS_MAX_OUTPUT_TOKENS_REASONING", 4096),
        log_dir=Path(os.getenv("HARNESS_LOG_DIR", "/tmp/harness_logs")),
    )
