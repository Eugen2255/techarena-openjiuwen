"""Точка входа харнесса.

Официальный запуск: python /app/run_agent.py --tasks-dir /data/tasks
"""
import sys

from harness.runner import main

if __name__ == "__main__":
    sys.exit(main())
