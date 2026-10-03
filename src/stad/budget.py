"""Выравнивание бюджета параметров — условие корректности сравнения.

Без этого шага эксперимент отвечает на вопрос «какая модель больше»,
а не «какая архитектура лучше». Трансформер с 4 головами и
гиперграф-энкодер при одинаковом ``hidden`` различаются по числу
параметров в разы, и любой вывод о превосходстве будет артефактом ёмкости.

Процедура: для каждой конфигурации ``(encoder, head)`` подбирается
``hidden`` так, чтобы итоговое число параметров попало в коридор
``target ± tolerance``. Подбор — бинарный поиск по сетке допустимых
значений ``hidden`` (кратных 8, чтобы multi-head attention делился).

Результат подбора записывается в отчёт: в статье приводится таблица
«конфигурация → hidden → фактическое число параметров», и читатель
видит, что сравнение честное.
"""
from __future__ import annotations

from dataclasses import dataclass

from .model import build_detector


@dataclass
class BudgetResult:
    """Что получилось при подборе."""

    hidden: int
    n_params: int
    target: int
    within_tolerance: bool
    tried: list[tuple[int, int]]

    @property
    def deviation(self) -> float:
        return abs(self.n_params - self.target) / max(1, self.target)


def count_params(
    encoder: str,
    head: str,
    *,
    hidden: int,
    n_features: int,
    n_nodes: int,
    window: int,
    encoder_kwargs: dict | None = None,
    head_kwargs: dict | None = None,
) -> int:
    det = build_detector(
        encoder, head, hidden=hidden, n_features=n_features, n_nodes=n_nodes,
        window=window, encoder_kwargs=encoder_kwargs, head_kwargs=head_kwargs,
    )
    return det.n_params


def match_budget(
    encoder: str,
    head: str,
    *,
    target: int,
    n_features: int,
    n_nodes: int,
    window: int,
    tolerance: float = 0.15,
    hidden_min: int = 16,
    hidden_max: int = 512,
    step: int = 8,
    encoder_kwargs: dict | None = None,
    head_kwargs: dict | None = None,
) -> BudgetResult:
    """Подобрать ``hidden`` под целевое число параметров.

    Монотонность числа параметров по ``hidden`` для всех наших
    архитектур позволяет использовать бинарный поиск; результат
    всё равно проверяется явно и помечается флагом
    ``within_tolerance``, чтобы несошедшиеся случаи были видны в отчёте,
    а не замаскированы.
    """
    grid = list(range(hidden_min, hidden_max + 1, step))
    tried: list[tuple[int, int]] = []

    def params_at(h: int) -> int:
        n = count_params(
            encoder, head, hidden=h, n_features=n_features, n_nodes=n_nodes,
            window=window, encoder_kwargs=encoder_kwargs, head_kwargs=head_kwargs,
        )
        tried.append((h, n))
        return n

    lo, hi = 0, len(grid) - 1
    best_i, best_gap = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = params_at(grid[mid])
        gap = abs(n - target)
        if gap < best_gap:
            best_i, best_gap = mid, gap
        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            best_i = mid
            break

    hidden = grid[best_i]
    n_params = dict(tried)[hidden]
    return BudgetResult(
        hidden=hidden,
        n_params=n_params,
        target=target,
        within_tolerance=abs(n_params - target) / max(1, target) <= tolerance,
        tried=sorted(tried),
    )
