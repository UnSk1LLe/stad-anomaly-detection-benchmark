"""Голова реконструкции — референсный механизм.

Самый распространённый подход в литературе по детекции аномалий трафика:
автоэнкодер учит восстанавливать норму, а ошибка восстановления служит
score. Так работают Con-GAE и MTGAE (Ren et al., 2024), и именно это
семейство бенчмарк FT-AED назвал наиболее перспективным.

Известная слабость, которую эксперимент должен вскрыть: достаточно
мощный декодер восстанавливает и аномалии тоже (обзор MTSAD прямо
отмечает, что реконструкционные методы переобучаются на аномалии).
Поэтому ёмкость декодера здесь сознательно ограничена — он
восстанавливает окно из одного вектора ``h``, без skip-соединений.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Head


class ReconHead(Head):
    """Декодирует ``h -> x̂`` целиком и считает MSE по окну.

    Score — средняя квадратичная ошибка по времени и признакам для
    каждого узла. Отсутствие skip-соединений принципиально: с ними
    модель копирует вход и score обнуляется для всего, включая аномалии.
    """

    needs_x = True

    def __init__(self, hidden: int, n_features: int, window: int, n_nodes: int, *, bottleneck: int | None = None) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        z = bottleneck or max(8, hidden // 4)
        self.encode = nn.Sequential(nn.Linear(hidden, z), nn.ReLU())
        self.decode = nn.Sequential(
            nn.Linear(z, hidden),
            nn.ReLU(),
            nn.Linear(hidden, window * n_features),
        )

    def _reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        out = self.decode(self.encode(h))
        return out.reshape(B, N, self.window, self.n_features)

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            raise ValueError("ReconHead требует x")
        return (self._reconstruct(h) - x).pow(2).mean()

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            raise ValueError("ReconHead требует x")
        return (self._reconstruct(h) - x).pow(2).mean(dim=(2, 3))
