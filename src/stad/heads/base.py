"""Интерфейс головы детекции — механизма, превращающего представление в score.

Это вторая ось эксперимента. Голова отвечает на вопрос «насколько это
представление невероятно», и способов ответить принципиально разных пять:

=====================  ==========================================================
Механизм               Что считается аномальным
=====================  ==========================================================
Реконструкция          то, что модель не может восстановить
Дискриминатор (GAN)    то, что критик отличает от нормы
Плотность (flow)       то, у чего низкое ``p(h)``
Остаток прогноза       то, что модель не предсказала
Контрастив             то, у чего расходятся представления разных видов
=====================  ==========================================================

Главная гипотеза бенчмарка: **при выровненном бюджете параметров выбор
механизма влияет сильнее, чем мощность энкодера.** Три косвенных
свидетельства за неё: DHMPN обходит SOTA-графы за счёт normalizing flow;
TransDe лидирует на MTSAD вообще без графа; PCA конкурентен с
OmniAnomaly без point-adjustment (Alves et al., 2026). Одно свидетельство
против, зато на целевом датасете: бенчмарк FT-AED (Coursey et al., 2024).
Эксперимент разрешает это противоречие.

Соглашение о знаке: **score тем больше, чем аномальнее**. Все метрики
полагаются на это.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn


class Head(nn.Module, ABC):
    """Базовая голова детекции.

    Attributes
    ----------
    splits_window:
        Если True, окно делится конвейером на вход энкодера и цель
        (``x[..., :-k, :]`` / ``x[..., -k:, :]``). Нужно головам,
        работающим на остатке прогноза: иначе цель попадает в энкодер
        и score вырождается в ноль.
    adversarial:
        Если True, обучение двухфазное: сначала шаг критика
        (:meth:`d_loss` по ``d_parameters``), затем шаг
        энкодера и головы (:meth:`loss` по ``main_parameters``).
    needs_x:
        Голове нужен исходный тензор окна, не только представление.
    """

    splits_window: bool = False
    adversarial: bool = False
    needs_x: bool = False

    def __init__(self, hidden: int, n_features: int, window: int, n_nodes: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_features = n_features
        self.window = window
        self.n_nodes = n_nodes

    @abstractmethod
    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Скалярная функция потерь для энкодера и головы."""

    @abstractmethod
    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """``-> [B, N]``, больше = аномальнее."""

    # --------------------------------------------------- параметры оптимизации
    def main_parameters(self) -> Iterator[nn.Parameter]:
        """Параметры, обновляемые основным оптимизатором."""
        return self.parameters()

    def d_parameters(self) -> Iterator[nn.Parameter]:
        """Параметры критика (только для ``adversarial``)."""
        return iter(())

    def d_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Потеря критика (только для ``adversarial``)."""
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def forecast_horizon(self) -> int:
        """Сколько последних шагов окна отрезается как цель прогноза."""
        return 0
