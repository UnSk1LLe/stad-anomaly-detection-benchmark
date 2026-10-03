"""Голова плотности: условный normalizing flow.

Мотивация — DHMPN («Efficient Directed Hypergraph Network for
Unsupervised Traffic Anomaly Detection»): представления моделируются
нормализующим потоком, что даёт **точную оценку плотности** и
вероятностный anomaly score вместо эвристической ошибки реконструкции.
Там же заявлено превосходство над SOTA графовыми моделями.

Принципиальное отличие от реконструкции и от критика: ``-log p(h)``
имеет ясный вероятностный смысл и калибруется — а калиброванная
уверенность нужна для safety-валидатора в контуре устранения. Ни
ошибка реконструкции, ни выход критика такого смысла не имеют.

Известная слабость, за которой надо следить в результатах:
оценка плотности чувствительна к сдвигу распределения (сезонность,
праздники, смена режима трафика). Поэтому в отчёте для этой головы
отдельно считается деградация между первой и второй половиной теста.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import Head


class _AffineCoupling(nn.Module):
    """Affine coupling: половина измерений преобразует другую половину.

    Логарифм якобиана при этом тривиален (сумма log-масштабов), что и
    даёт точное правдоподобие без аппроксимаций.
    """

    def __init__(self, dim: int, hidden: int, *, flip: bool) -> None:
        super().__init__()
        self.flip = flip
        self.d_keep = dim // 2
        self.d_mod = dim - self.d_keep
        self.net = nn.Sequential(
            nn.Linear(self.d_keep, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * self.d_mod),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = (z[..., self.d_keep:], z[..., : self.d_keep]) if self.flip else (
            z[..., : self.d_keep], z[..., self.d_keep:]
        )
        shift, log_scale = self.net(a).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 2.0          # стабилизация
        b = b * log_scale.exp() + shift
        out = torch.cat([b, a] if self.flip else [a, b], dim=-1)
        return out, log_scale.sum(-1)


class FlowHead(Head):
    """``score = -log p(h)`` под обученным нормализующим потоком.

    Обучение — прямая максимизация правдоподобия на нормальных окнах;
    никакого состязательного процесса и никакого декодера.
    """

    needs_x = False

    def __init__(
        self,
        hidden: int,
        n_features: int,
        window: int,
        n_nodes: int,
        *,
        n_coupling: int = 4,
        flow_hidden: int | None = None,
    ) -> None:
        super().__init__(hidden, n_features, window, n_nodes)
        fh = flow_hidden or max(16, hidden // 2)
        self.pre = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.couplings = nn.ModuleList(
            _AffineCoupling(hidden, fh, flip=bool(i % 2)) for i in range(n_coupling)
        )
        const = 0.5 * hidden * float(np.log(2.0 * np.pi))
        self.register_buffer("const", torch.tensor(const), persistent=False)

    def _neg_log_prob(self, h: torch.Tensor) -> torch.Tensor:
        z = self.pre(h)
        logdet = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
        for layer in self.couplings:
            z, ld = layer(z)
            logdet = logdet + ld
        # -log p(h) = -[log N(z;0,I) + logdet]
        return 0.5 * z.pow(2).sum(-1) + self.const - logdet

    def loss(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        return self._neg_log_prob(h).mean()

    def score(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        return self._neg_log_prob(h)
