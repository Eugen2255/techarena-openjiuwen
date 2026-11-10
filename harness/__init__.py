"""Харнесс кодового агента на openJiuwen для TechArena 2026."""
import os
import tempfile

try:
    from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG
    from openjiuwen.core.common.logging.log_config import configure_log_config
except ImportError:  # тесты быстрого пути (tests/) запускаются и без openJiuwen
    configure_log_config = None


def _configure_openjiuwen_logging() -> None:
    """openJiuwen по умолчанию пишет логи в ./logs и шумит на уровне INFO.

    В контейнере /app доступна только для чтения, поэтому до первого использования логгеров
    оставляем только консоль, а путь для файлов на всякий случай переносим в /tmp.
    """
    config = dict(DEFAULT_INNER_LOG_CONFIG)
    config.update(
        level=os.getenv("HARNESS_OPENJIUWEN_LOG_LEVEL", "WARNING"),
        log_path=os.path.join(tempfile.gettempdir(), "openjiuwen_logs", ""),
        output=["console"],
        interface_output=["console"],
        performance_output=["console"],
        propagate=False,
        # Логгер llm при каждой ошибке вызова (например, 429) печатает весь промпт; повторы логирует harness.llm.
        loggers={"llm": {"level": "CRITICAL"}},
    )
    configure_log_config(config)


if configure_log_config is not None:
    _configure_openjiuwen_logging()
