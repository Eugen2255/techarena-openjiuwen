# Харнесс TechArena на openJiuwen.
# Запуск у организаторов: read-only FS, писать можно только в /tmp и /data/tasks, сети нет (кроме прокси LLM).
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Для локальной сборки можно указать зеркало (scripts/run_dev.py берёт его из DEV_PIP_INDEX_URL).
# У организаторов аргумент не передаётся, пакеты ставятся с PyPI.
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.txt

# tiktoken скачивает словари при первом использовании, а при запуске сети нет: кладём их в образ.
RUN python -c "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base', 'o200k_base')]"

COPY run_agent.py ./
COPY harness ./harness
# Байткод собираем при сборке: при запуске /app доступен только для чтения.
RUN python -m compileall -q /app/harness /app/run_agent.py

ENTRYPOINT ["python", "/app/run_agent.py"]
