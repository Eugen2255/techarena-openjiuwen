"""Подключение моделей через openJiuwen: учёт токенов, лимит частоты и повторы при временных ошибках."""
from __future__ import annotations

import asyncio
import logging
import random
import re

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

from harness.config import Settings, is_reasoning
from harness.meter import INPUT_WEIGHT, BudgetExceeded, TokenMeter

log = logging.getLogger(__name__)

# openJiuwen вызывает модель без повторов (max_retries=0), поэтому повторяем сами.
MAX_RETRIES = 3
# Сколько ждать, пока параллельные вызовы освободят резерв бюджета.
MAX_RESERVE_WAIT_S = 120
# Временные ошибки: лимит частоты, ошибки сервера, таймауты, обрыв соединения.
_RETRYABLE_RE = re.compile(
    r"RateLimitError|Too Many Requests|\b429\b|InternalServerError|\b50[0234]\b|"
    r"APITimeoutError|timed out|APIConnectionError|Connection error",
    re.IGNORECASE,
)
# Таймаут: модель могла сгенерировать ответ до обрыва, и прокси посчитает его выход.
_TIMEOUT_RE = re.compile(r"APITimeoutError|timed out|Timeout", re.IGNORECASE)


class RateLimiter:
    """Лимиты NeuralDeep на весь прогон: не больше rpm запросов в минуту и не больше concurrency одновременно."""

    def __init__(self, rpm: float, concurrency: int):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(max(1, concurrency))

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            delay = self._next - now
            self._next = max(now, self._next) + self._interval
        if delay > 0:
            await asyncio.sleep(delay)


_limiter: RateLimiter | None = None


def _get_limiter(settings: Settings) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(settings.requests_per_minute, settings.llm_concurrency)
    return _limiter


def _error_chain(exc: BaseException) -> str:
    texts = []
    while exc is not None and len(texts) < 5:
        texts.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or exc.__context__
    return " | ".join(texts)


def _is_retryable(exc: BaseException) -> bool:
    return bool(_RETRYABLE_RE.search(_error_chain(exc)))


def _short_reason(exc: BaseException) -> str:
    """Короткая причина для лога: 429, 5xx, таймаут — openJiuwen оборачивает их в FrameworkError."""
    chain = _error_chain(exc)
    match = _RETRYABLE_RE.search(chain)
    return chain[max(0, match.start() - 20):match.end() + 100] if match else chain[:120]


class MeteredModel(Model):
    """Model из openJiuwen, который считает токены каждого вызова и не пускает вызовы сверх бюджета."""

    def __init__(self, *, meter: TokenMeter, task_id: str, limiter: RateLimiter, max_output_tokens: int,
                 model_client_config: ModelClientConfig, model_config: ModelRequestConfig):
        super().__init__(model_client_config=model_client_config, model_config=model_config)
        self._meter = meter
        self._task_id = task_id
        self._limiter = limiter
        self._max_output_tokens = max_output_tokens

    @staticmethod
    def _approx_input_tokens(args, kwargs) -> float:
        """Вход по длине сообщений: ~3 символа на токен, с запасом на описание инструментов."""
        messages = args[0] if args else kwargs.get("messages", "")
        return len(str(messages)) / 3 + 500

    def _worst_case_cost(self, args, kwargs) -> float:
        """Сверху оценивает стоимость вызова: оценка входа плюс полный потолок выхода."""
        return self._max_output_tokens + self._approx_input_tokens(args, kwargs) * INPUT_WEIGHT

    async def _reserve(self, worst_case: float) -> None:
        """Резервирует бюджет на вызов. Если мешают только резервы идущих вызовов, ждёт их окончания."""
        waited = 0.0
        while True:
            try:
                self._meter.reserve(self._task_id, worst_case)
                return
            except BudgetExceeded:
                if not self._meter.fits(self._task_id, worst_case) or waited >= MAX_RESERVE_WAIT_S:
                    raise
            await asyncio.sleep(0.5)
            waited += 0.5

    async def invoke(self, *args, **kwargs):
        worst_case = self._worst_case_cost(args, kwargs)
        for attempt in range(MAX_RETRIES + 1):
            await self._reserve(worst_case)
            try:
                async with self._limiter.slots:
                    await self._limiter.wait()
                    message = await super().invoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — решаем ниже, повторять или пробрасывать
                # Ушедший запрос прокси считает и при ошибке: списываем оценку входа, при таймауте — и потолок выхода.
                timeout = bool(_TIMEOUT_RE.search(_error_chain(exc)))
                self._meter.charge(self._task_id, self._approx_input_tokens(args, kwargs),
                                   self._max_output_tokens if timeout else 0)
                if attempt == MAX_RETRIES or not _is_retryable(exc):
                    raise
                delay = 2 ** attempt + random.random()
                log.warning("[%s] LLM call failed (%s), retry %d/%d in %.1fs",
                            self._task_id, _short_reason(exc), attempt + 1, MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                continue
            finally:
                self._meter.release(worst_case)
            self._meter.record(self._task_id, getattr(message, "usage_metadata", None))
            return message

    async def stream(self, *args, **kwargs):
        # Стриминг агент сейчас не использует; повторов здесь нет, только бюджет и лимиты NeuralDeep.
        worst_case = self._worst_case_cost(args, kwargs)
        await self._reserve(worst_case)
        usage = None
        try:
            async with self._limiter.slots:
                await self._limiter.wait()
                async for chunk in super().stream(*args, **kwargs):
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    yield chunk
        finally:
            self._meter.release(worst_case)
        self._meter.record(self._task_id, usage)


def client_config(settings: Settings) -> ModelClientConfig:
    return ModelClientConfig(
        client_provider="OpenAI",
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        verify_ssl=settings.verify_ssl,
    )


def max_output_tokens(settings: Settings, model_name: str) -> int:
    return settings.max_output_tokens_reasoning if is_reasoning(model_name) else settings.max_output_tokens


def request_config(settings: Settings, model_name: str) -> ModelRequestConfig:
    return ModelRequestConfig(model=model_name, temperature=0.0, max_tokens=max_output_tokens(settings, model_name))


def make_model(settings: Settings, meter: TokenMeter, task_id: str, model_name: str) -> MeteredModel:
    return MeteredModel(
        meter=meter,
        task_id=task_id,
        limiter=_get_limiter(settings),
        max_output_tokens=max_output_tokens(settings, model_name),
        model_client_config=client_config(settings),
        model_config=request_config(settings, model_name),
    )
