"""Голова на основе критика (BiGAN) — и её теоретическая проблема.

Постановка следует CCB-GraphGAN (Nouri et al., 2026), который на FT-AED
достигает обнаружения ДТП в среднем на 5 минут раньше официального
отчёта при FPR 1%: двунаправленное отображение ``E: h -> z`` и
``G: z -> ĥ`` с цикл-согласованностью, критик ``D(h, z)``.

**Открытый пробел, который эта голова проверяет эмпирически.** В
равновесии идеального GAN ``D* = 1/2`` для всех входов, то есть выход
критика не несёт информации об аномалии; WGAN-критик — не плотность.
Поэтому любой score, опирающийся только на ``D``, теоретически не
обоснован. Практический компромисс, реализованный здесь, — смесь
ошибки цикла и выхода критика с коэффициентом ``lam``:

.. math::
    s(h) = \\|h - G(E(h))\\|^2 + \\lambda\\,(1 - D(h, E(h)))

Если в эксперименте оптимальное ``lam`` окажется близким к нулю, это
прямое свидетельство, что вклад даёт цикл-реконструкция, а не критик, —
и тогда «GAN-детектор» фактически является автоэнкодером с
состязательной регуляризацией. Такой результат сам по себе публикуем.
"""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from .base import Head


class BiGANHead(Head):
    """Цикл-согласованный BiGAN поверх представления узла."""

    adversarial = True
    needs_x = False

    def __init__(
        self,
        hidden: int,
        n_features: int,
        window: int,
        n_nodes: int,
        *,
        latent: int | None = None,
        lam: float = 0.1,
        cycle_weight: float = 1.0,
        label_smooth: float = 0.9,
    ) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        z = latent or max(8, hidden // 4)
        self.lam = lam
        self.cycle_weight = cycle_weight
        self.label_smooth = label_smooth
        self.latent = z

        self.E = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, z))
        self.G = nn.Sequential(nn.Linear(z, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.D = nn.Sequential(
            nn.Linear(hidden + z, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    # ------------------------------------------------------------- параметры
    def main_parameters(self) -> Iterator[nn.Parameter]:
        """Критик обновляется отдельно, поэтому исключён."""
        d_ids = {id(p) for p in self.D.parameters()}
        return (p for p in self.parameters() if id(p) not in d_ids)

    def d_parameters(self) -> Iterator[nn.Parameter]:
        return self.D.parameters()

    # ----------------------------------------------------------------- логика
    def _pairs(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_real = self.E(h)                                   # (h, E(h))  — «реальная» пара
        z_fake = torch.randn_like(z_real)
        h_fake = self.G(z_fake)                              # (G(z), z)  — «фейковая» пара
        return z_real, h_fake, z_fake, self.G(z_real)        # последнее — цикл

    def d_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Критик учит отличать ``(h, E(h))`` от ``(G(z), z)``."""
        h = h.detach()
        z_real, h_fake, z_fake, _ = self._pairs(h)
        logit_real = self.D(torch.cat([h, z_real.detach()], -1))
        logit_fake = self.D(torch.cat([h_fake.detach(), z_fake], -1))
        bce = nn.functional.binary_cross_entropy_with_logits
        return bce(logit_real, torch.full_like(logit_real, self.label_smooth)) + bce(
            logit_fake, torch.zeros_like(logit_fake)
        )

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        """E и G обманывают критика плюс держат цикл-согласованность."""
        z_real, h_fake, z_fake, h_cycle = self._pairs(h)
        logit_real = self.D(torch.cat([h, z_real], -1))
        logit_fake = self.D(torch.cat([h_fake, z_fake], -1))
        bce = nn.functional.binary_cross_entropy_with_logits
        adv = bce(logit_real, torch.zeros_like(logit_real)) + bce(
            logit_fake, torch.ones_like(logit_fake)
        )
        cycle = (h_cycle - h).pow(2).mean()
        return adv + self.cycle_weight * cycle

    @torch.no_grad()
    def _score_parts(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.E(h)
        cycle = (self.G(z) - h).pow(2).mean(-1)
        critic = 1.0 - torch.sigmoid(self.D(torch.cat([h, z], -1))).squeeze(-1)
        return cycle, critic

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        cycle, critic = self._score_parts(h)
        return cycle + self.lam * critic

    @torch.no_grad()
    def score_components(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Отдельно цикл и критик — для абляции вклада ``D``.

        Используется в ``scripts/run_grid.py --ablate-bigan``: если
        ранжирование по ``cycle`` не хуже, чем по полному score, значит
        критик не несёт информации об аномалии, как и предсказывает
        теория ``D* = 1/2``.
        """
        cycle, critic = self._score_parts(h)
        return {"cycle": cycle, "critic": critic, "combined": cycle + self.lam * critic}
