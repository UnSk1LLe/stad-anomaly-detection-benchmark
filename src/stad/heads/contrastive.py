"""Контрастивная голова: расхождение представлений разных видов окна.

Механизм TransDe («Decomposition-based multi-scale transformer framework
for time series anomaly detection», Neural Networks 2025): никакой
реконструкции. Из окна строятся два вида разного масштаба, представления
выравниваются контрастивно на нормальных данных, а anomaly score — это
расхождение между видами. На MTSAD такой подход лидирует на четырёх
бенчмарках из пяти, **и при этом вообще не использует граф** — что и
делает его важным для нашей гипотезы о приоритете механизма над энкодером.

Логика: на нормальном окне крупный и мелкий масштаб согласованы (тренд
объясняет сезонность); на аномальном — расходятся. Stop-gradient на одной
ветви предотвращает коллапс в постоянную функцию.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Head


class ContrastiveHead(Head):
    """Два проекционных вида представления + симметричное KL-расхождение."""

    needs_x = True

    def __init__(
        self,
        hidden: int,
        n_features: int,
        window: int,
        n_nodes: int,
        *,
        proj: int | None = None,
        patch: int = 4,
        temperature: float = 0.5,
    ) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        p = proj or max(8, hidden // 2)
        self.patch = max(2, min(patch, window))
        self.temperature = temperature

        # ветвь «мелкий масштаб»: статистики по патчам исходного окна
        self.fine = nn.Sequential(nn.Linear(2 * n_features, p), nn.ReLU(), nn.Linear(p, p))
        # ветвь «крупный масштаб»: представление энкодера
        self.coarse = nn.Sequential(nn.Linear(hidden, p), nn.ReLU(), nn.Linear(p, p))

    def _views(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T, Fd = x.shape
        k = self.patch
        n_patch = T // k
        if n_patch < 1:
            patches = x.unsqueeze(2)
        else:
            patches = x[:, :, : n_patch * k].reshape(B, N, n_patch, k, Fd)
        stats = torch.cat([patches.mean(3), patches.std(3).clamp(min=1e-6)], dim=-1)  # [B,N,P,2F]
        fine = F.log_softmax(self.fine(stats).mean(2) / self.temperature, dim=-1)
        coarse = F.log_softmax(self.coarse(h) / self.temperature, dim=-1)
        return fine, coarse

    @staticmethod
    def _sym_kl(a_log: torch.Tensor, b_log: torch.Tensor) -> torch.Tensor:
        """Симметричная KL между двумя лог-распределениями, поэлементно по узлам."""
        a, b = a_log.exp(), b_log.exp()
        return ((a * (a_log - b_log)).sum(-1) + (b * (b_log - a_log)).sum(-1)) * 0.5

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Сблизить виды на норме; stop-gradient по очереди на каждой ветви."""
        if x is None:
            raise ValueError("ContrastiveHead требует x")
        fine, coarse = self._views(h, x)
        d1 = self._sym_kl(fine, coarse.detach())
        d2 = self._sym_kl(fine.detach(), coarse)
        return (d1 + d2).mean() * 0.5

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            raise ValueError("ContrastiveHead требует x")
        fine, coarse = self._views(h, x)
        return self._sym_kl(fine, coarse)
