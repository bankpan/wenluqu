from __future__ import annotations

import re
from dataclasses import dataclass

SCORE_RANGE_PATTERN = re.compile(r"(?P<lo>\d{3})\D+(?P<hi>\d{3})")
INTEGER_PATTERN = re.compile(r"\d+")


class RowValidationError(ValueError):
    pass


@dataclass(slots=True)
class ScoreRow:
    score_range: str
    lower: int
    upper: int
    candidates: int
    admitted: int


def parse_score_range(text: str) -> tuple[int, int]:
    match = SCORE_RANGE_PATTERN.search(text)
    if not match:
        raise RowValidationError(f"无法解析分数段: '{text}'")
    lo = int(match.group("lo"))
    hi = int(match.group("hi"))
    if lo >= hi:
        raise RowValidationError(f"分数下限必须小于上限: '{text}'")
    return lo, hi


def parse_integer(text: str) -> int:
    digits = INTEGER_PATTERN.findall(text)
    if not digits:
        raise RowValidationError(f"无法解析数字: '{text}'")
    return int("".join(digits))


def validate_row(cells: list[str]) -> ScoreRow:
    if len(cells) < 3:
        raise RowValidationError(f"列数不足: {cells}")

    clean_cells = [str(cell).strip() for cell in cells if str(cell).strip()]
    if len(clean_cells) < 3:
        raise RowValidationError(f"有效列数不足: {cells}")

    score_cell = next((cell for cell in clean_cells if SCORE_RANGE_PATTERN.search(cell)), clean_cells[0])
    lo, hi = parse_score_range(score_cell)

    numbers: list[int] = []
    score_consumed = False
    for cell in clean_cells:
        if not score_consumed and cell == score_cell:
            score_consumed = True
            continue
        if SCORE_RANGE_PATTERN.search(cell):
            continue
        if INTEGER_PATTERN.search(cell):
            numbers.append(parse_integer(cell))

    if len(numbers) < 2:
        raise RowValidationError(f"未找到足够的数量列: {cells}")

    candidates, admitted = numbers[0], numbers[1]
    if admitted > candidates:
        raise RowValidationError(f"录取人数({admitted})大于复试人数({candidates})")

    return ScoreRow(
        score_range=score_cell,
        lower=lo,
        upper=hi,
        candidates=candidates,
        admitted=admitted,
    )
