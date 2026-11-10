"""data_analysis: сумма units по товару и диапазону месяцев в CSV.

Шаблон из примеров:
  The file sales.csv has columns order_id,region,product,month,units,price (20 rows).
  How many units of product 'widget' were sold in months 7 through 8 inclusive? ...
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from harness.fastpath import FastAnswer, register
from harness.tasks import Task

_FILE_RE = re.compile(r"The file (?P<file>[\w.-]+\.csv) has columns (?P<columns>[\w,]+)")
_QUESTION_RE = re.compile(
    r"How many units of product '(?P<product>[^']+)' were sold in months "
    r"(?P<first>\d+) through (?P<last>\d+) inclusive\?"
)
_REQUIRED = {"product", "month", "units"}


@register("data_analysis")
def referenced_file_missing(task: Task, workdir: Path) -> FastAnswer | None:
    """Инструкция ссылается на файл с данными, которого нет в папке задачи: посчитать нечего — невыполнима."""
    file_match = re.search(r"The file (?P<file>[\w.-]+\.(?:csv|tsv|json|txt|xlsx))\b", task.instruction)
    if not file_match or not workdir.is_dir():
        return None
    name = file_match["file"]
    if (workdir / name).is_file() or any(p.name == name for p in workdir.rglob("*")):
        return None
    return FastAnswer(None, infeasible=True, note=f"referenced_file_missing: {name}")


@register("data_analysis")
def units_by_product_and_months(task: Task, workdir: Path) -> FastAnswer | None:
    file_match = _FILE_RE.search(task.instruction)
    question = _QUESTION_RE.search(task.instruction)
    if not file_match or not question:
        return None
    first, last = int(question["first"]), int(question["last"])
    if not 1 <= first <= last <= 12:
        return None

    path = workdir / file_match["file"]
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not _REQUIRED <= set(reader.fieldnames or ()):
            return None
        rows = list(reader)

    total = 0
    found_product = False
    for row in rows:
        try:
            month, units = int(row["month"]), int(row["units"])
        except (TypeError, ValueError):
            return None  # грязные данные: пусть разбирается агент
        if row["product"] == question["product"]:
            found_product = True
            if first <= month <= last:
                total += units
    if not found_product:
        # TODO: такого товара нет — это ответ 0 или невыполнимая задача? Пока отдаём агенту.
        return None
    return FastAnswer(str(total), note="units_by_product_and_months")
