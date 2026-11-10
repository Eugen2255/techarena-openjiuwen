#!/usr/bin/env python3
"""Локальный прогон харнесса на копии задач и подсчёт результата.

    python scripts/run_dev.py                       # Docker с флагами как у организаторов (основной способ)
    python scripts/run_dev.py --only math_0000      # только выбранные задачи
    python scripts/run_dev.py --tasks DIR           # другой набор задач (по умолчанию tasks_examples)
    python scripts/run_dev.py --local               # без Docker, в текущем Python (Linux/macOS/WSL)

Ключи берутся из .env в корне репозитория. Результаты — в runs/<время>/.
Нужен только стандартный Python 3.10+ и Docker; зависимости харнесса ставятся в образ.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGE = "hack:dev"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    # utf-8-sig: Блокнот может сохранить .env с BOM, и тогда имя первой переменной было бы испорчено.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def copy_tasks(src: Path, dst: Path, only: list[str]) -> None:
    dst.mkdir(parents=True)
    for task_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        if only and task_dir.name not in only:
            continue
        # hidden/ — скрытые тесты генератора: в контейнер их не отдаём, evaluate.py берёт их из исходного набора.
        shutil.copytree(task_dir, dst / task_dir.name,
                        ignore=shutil.ignore_patterns("answer.txt", "__pycache__", ".DS_Store", "hidden"))


def docker_command(tasks_dir: Path, logs_dir: Path, name: str) -> list[str]:
    """Повторяет команду из регламента. Отличия: нет --network hack-net, доп. папка для логов и HARNESS_*."""
    extra_env = ["-e", "BUDGET"] if os.getenv("BUDGET") else []
    for key in sorted(os.environ):
        if key.startswith("HARNESS_") and key != "HARNESS_LOG_DIR":
            extra_env += ["-e", key]
    return [
        "docker", "run", "--rm", "--name", name,
        "-e", "LLM_BASE_URL", "-e", "LLM_API_KEY", *extra_env,
        "-v", f"{tasks_dir}:/data/tasks:rw",
        "-v", f"{logs_dir}:/data/logs:rw", "-e", "HARNESS_LOG_DIR=/data/logs",
        "--read-only", "--tmpfs", "/tmp:rw,size=2g", "-e", "HOME=/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", str(min(8, os.cpu_count() or 1)), "--memory", "16g", "--pids-limit", "512",
        "--workdir", "/app", "--entrypoint", "python",
        IMAGE, "/app/run_agent.py", "--tasks-dir", "/data/tasks",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=REPO / "tasks_examples", help="исходный набор задач")
    parser.add_argument("--only", nargs="*", default=[], help="id задач (имена папок)")
    parser.add_argument("--local", action="store_true", help="запуск без Docker")
    parser.add_argument("--no-build", action="store_true", help="не пересобирать образ")
    parser.add_argument("--timeout", type=int, default=3700, help="таймаут прогона, секунд")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    if not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_API_KEY"):
        print("Задайте LLM_BASE_URL и LLM_API_KEY в .env (см. .env.example)", file=sys.stderr)
        return 2

    run_dir = REPO / "runs" / time.strftime("%Y%m%d_%H%M%S")
    tasks_dir, logs_dir = run_dir / "tasks", run_dir / "logs"
    copy_tasks(args.tasks.resolve(), tasks_dir, args.only)
    logs_dir.mkdir(parents=True)
    # Исходный набор нужен scripts/fit_estimator.py: признаки считаются по задачам до решения.
    (run_dir / "source.txt").write_text(str(args.tasks.resolve()), encoding="utf-8")
    print(f"run dir: {run_dir}", flush=True)

    if args.local:
        env = {**os.environ, "HARNESS_LOG_DIR": str(logs_dir)}
        cmd = [sys.executable, str(REPO / "run_agent.py"), "--tasks-dir", str(tasks_dir),
               "--out", str(run_dir / "solutions.json")]
    else:
        env = None
        if not args.no_build:
            build = ["docker", "build", "-t", IMAGE]
            if os.getenv("DEV_PIP_INDEX_URL"):
                build += ["--build-arg", f"PIP_INDEX_URL={os.environ['DEV_PIP_INDEX_URL']}"]
            subprocess.run([*build, str(REPO)], check=True)
        cmd = docker_command(tasks_dir, logs_dir, f"run-dev-{run_dir.name}")

    started = time.monotonic()
    try:
        code = subprocess.run(cmd, env=env, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"harness exit code: {code}, {time.monotonic() - started:.0f}s", flush=True)

    return subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), str(tasks_dir),
                           "--original", str(args.tasks.resolve()), "--logs", str(logs_dir)]).returncode


if __name__ == "__main__":
    sys.exit(main())
