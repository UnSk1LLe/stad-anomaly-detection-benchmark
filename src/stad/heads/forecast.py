"""Голова остатка прогноза — и голова physics-informed остатка.

Два механизма, объединённые одним принципом: аномалия есть расхождение
между предсказанным и наблюдённым. Различие — откуда берётся предсказание.

* :class:`ForecastHead` — предсказание выучено из данных. Вероятностный
  вариант (μ, σ) позволяет нормировать остаток на собственную
  неопределённость модели, иначе шумные узлы дают ложные тревоги.
* :class:`PhysicsResidualHead` — предсказание даёт **закон сохранения**
  транспортного потока, а не нейросеть.

Почему вторая важна для диссертации. Это единственный механизм в сетке,
чей score **не зависит от обученного генератора**, поэтому он
структурно невосприимчив к пробелу «циркулярность оценки»: его нельзя
обмануть, подсунув артефакты генератора. В литературе physics-informed
подход применён к обнаружению нерекуррентных заторов (PINN на Seoul Ring
Expressway) и к генерации, но **не как голова детектора поверх
обучаемого энкодера** — это незанятая ниша.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Head


class ForecastHead(Head):
    """Вероятностный прогноз на ``horizon`` шагов; score — нормированный остаток.

    Окно режется конвейером: энкодер видит ``x[..., :-horizon, :]``,
    голова предсказывает ``x[..., -horizon:, :]``. Без этого разделения
    цель утекает в представление и score вырождается.
    """

    splits_window = True
    needs_x = True

    def __init__(
        self,
        hidden: int,
        n_features: int,
        window: int,
        n_nodes: int,
        *,
        horizon: int = 3,
        min_sigma: float = 1e-2,
    ) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        self.horizon = horizon
        self.min_sigma = min_sigma
        out = horizon * n_features
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.log_sigma = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out))

    @property
    def forecast_horizon(self) -> int:
        return self.horizon

    def _predict(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape
        shape = (B, N, self.horizon, self.n_features)
        mu = self.mu(h).reshape(shape)
        sigma = self.log_sigma(h).clamp(-6.0, 4.0).exp().reshape(shape) + self.min_sigma
        return mu, sigma

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Отрицательное гауссово log-правдоподобие цели."""
        if x is None:
            raise ValueError("ForecastHead требует целевое окно x")
        mu, sigma = self._predict(h)
        return (((x - mu) / sigma).pow(2) * 0.5 + sigma.log()).mean()

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Z-нормированный остаток: сюрприз относительно своей уверенности."""
        if x is None:
            raise ValueError("ForecastHead требует целевое окно x")
        mu, sigma = self._predict(h)
        return (((x - mu) / sigma).pow(2)).mean(dim=(2, 3))


class PhysicsResidualHead(Head):
    """Невязка уравнения сохранения LWR как anomaly score.

    Для одномерного потока закон сохранения:

    .. math::
        \\frac{\\partial \\rho}{\\partial t} + \\frac{\\partial q}{\\partial x} = 0

    где :math:`\\rho` — плотность (пропорциональна занятости), :math:`q` —
    поток. В нормальном трафике невязка близка к нулю с точностью до шума
    детекторов; ДТП, перекрытие полосы и отказ датчика нарушают баланс.

    Обучаемая часть минимальна и сознательно: калибровочный масштаб на
    признак плюс скаляр, взвешивающий обученную поправку от энкодера.
    Если в результатах поправка окажется нужной (вес заметно больше нуля),
    это говорит, что чистой физики недостаточно; если нет — физика
    работает сама и даёт generator-independent детектор почти без
    параметров, что исключительно ценно для edge-развёртывания.

    Требует, чтобы порядок узлов соответствовал порядку вдоль коридора
    (это обеспечивает ``data.ft_aed.corridor_adjacency``) и чтобы в
    признаках были занятость и объём.
    """

    needs_x = True

    def __init__(
        self,
        hidden: int,
        n_features: int,
        window: int,
        n_nodes: int,
        *,
        occ_idx: int = 1,
        vol_idx: int = 2,
        lanes: int = 3,
        learn_correction: bool = True,
    ) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        self.occ_idx = occ_idx
        self.vol_idx = vol_idx
        self.lanes = lanes
        self.scale = nn.Parameter(torch.ones(2))
        self.correction_weight = nn.Parameter(torch.zeros(1))
        self.correction = (
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            if learn_correction
            else None
        )

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,N,T,F] -> [B,N]`` средняя |невязка| по времени."""
        rho = x[..., self.occ_idx] * self.scale[0]            # [B,N,T]
        q = x[..., self.vol_idx] * self.scale[1]

        d_rho_dt = rho[..., 1:] - rho[..., :-1]              # [B,N,T-1]

        # производная по пространству: вдоль коридора на фиксированной полосе,
        # то есть шаг по узлам равен числу полос
        L = self.lanes
        dq_dx = torch.zeros_like(q)
        if q.shape[1] > L:
            dq_dx[:, L:] = q[:, L:] - q[:, :-L]
            dq_dx[:, :L] = dq_dx[:, L : 2 * L]               # край: повтор соседа
        dq_dx = dq_dx[..., 1:]

        return (d_rho_dt + dq_dx).abs().mean(-1)             # [B,N]

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """Калибровка: на нормальных окнах невязка должна быть минимальной."""
        if x is None:
            raise ValueError("PhysicsResidualHead требует x")
        base = self._residual(x)
        if self.correction is not None:
            base = base + self.correction_weight * self.correction(h).squeeze(-1)
        # L1 по невязке + слабая регуляризация масштабов к единице
        return base.abs().mean() + 1e-3 * (self.scale - 1.0).pow(2).sum()

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            raise ValueError("PhysicsResidualHead требует x")
        base = self._residual(x)
        if self.correction is not None:
            base = base + self.correction_weight * self.correction(h).squeeze(-1)
        return base

    @torch.no_grad()
    def correction_share(self, h: torch.Tensor, x: torch.Tensor) -> float:
        """Доля score, объяснённая обученной поправкой (для отчёта)."""
        if self.correction is None:
            return 0.0
        phys = self._residual(x).abs().mean().item()
        corr = (self.correction_weight * self.correction(h).squeeze(-1)).abs().mean().item()
        return float(corr / (phys + corr + 1e-12))
