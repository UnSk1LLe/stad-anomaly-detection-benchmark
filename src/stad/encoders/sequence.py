"""Неграфовые энкодеры — проверка вклада пространственного prior.

Обе архитектуры взяты из контролируемого бенчмарка Dong et al. (2026) на
METR-LA, где Transformer, STGCN, DCRNN и Gated TCN сравнивались в едином
режиме. Здесь они играют роль нижней границы по пространственности:

* :class:`GatedTCN` — чистая временная свёртка, узлы независимы. Если
  графовые модели не отрываются от него статистически значимо, весь
  графовый аппарат в этой задаче не нужен.
* :class:`NodeTransformer` — полное self-attention по узлам, без маски
  графа. Проверяет обратную гипотезу: достаточно ли внимания, чтобы
  восстановить нужные связи самостоятельно, без физического prior.
  Обзорная литература приписывает трансформерам превосходство в
  темпоральном контексте (Pérez et al., 2025), а для incident detection
  превосходство показано прямо (Lu et al., 2024).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import Encoder


class GatedTCN(Encoder):
    """Dilated temporal conv с gated-активацией; узлы обрабатываются независимо.

    Экспоненциально растущая дилатация даёт рецептивное поле, покрывающее
    окно целиком, при логарифмическом числе слоёв — поэтому TCN честно
    конкурирует с рекуррентными моделями, а не является «слабым» бейзлайном.
    """

    uses_graph = False

    def __init__(self, n_features: int, hidden: int, n_nodes: int, window: int, *, layers: int = 3) -> None:
        super().__init__(n_features, hidden, n_nodes, window)
        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residual = nn.ModuleList()
        c_in = n_features
        for i in range(layers):
            d = 2**i
            self.layers.append(nn.Conv1d(c_in, hidden, 3, padding=d, dilation=d))
            self.gates.append(nn.Conv1d(c_in, hidden, 3, padding=d, dilation=d))
            self.residual.append(nn.Conv1d(c_in, hidden, 1) if c_in != hidden else nn.Identity())
            c_in = hidden
        self.norm = nn.LayerNorm(hidden)

    def set_graph(self, A: torch.Tensor) -> None:  # noqa: ARG002
        """Граф не используется — в этом и смысл архитектуры."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check(x)
        B, N, T, F = x.shape
        z = x.reshape(B * N, T, F).transpose(1, 2)            # [BN, F, T]
        for conv, gate, res in zip(self.layers, self.gates, self.residual):
            h = torch.tanh(conv(z)[..., :T]) * torch.sigmoid(gate(z)[..., :T])
            z = h + res(z)[..., :T]
        out = z[..., -1].reshape(B, N, self.hidden)
        return torch.relu(self.norm(out))


class _PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class NodeTransformer(Encoder):
    """Temporal self-attention, затем self-attention по узлам без маски графа.

    Две стадии внимания нужны, чтобы архитектура отличалась от GAT-LSTM
    ровно одним — отсутствием физической маски. Если NodeTransformer
    обходит GAT-LSTM, значит prior коридора не нужен; если проигрывает —
    prior несёт информацию, которую внимание не восстанавливает из
    ограниченной обучающей выборки.
    """

    uses_graph = False

    def __init__(
        self,
        n_features: int,
        hidden: int,
        n_nodes: int,
        window: int,
        *,
        n_heads: int = 4,
        ff_mult: int = 2,
    ) -> None:
        super().__init__(n_features, hidden, n_nodes, window)
        heads = max(1, min(n_heads, hidden // 8))
        while hidden % heads:
            heads -= 1
        self.proj = nn.Linear(n_features, hidden)
        self.pos = _PositionalEncoding(hidden)
        self.temporal = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=hidden * ff_mult,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.spatial = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=hidden * ff_mult,
            batch_first=True, dropout=0.1, norm_first=True,
        )
        self.norm = nn.LayerNorm(hidden)

    def set_graph(self, A: torch.Tensor) -> None:  # noqa: ARG002
        """Маска графа сознательно не применяется."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check(x)
        B, N, T, F = x.shape
        z = self.pos(self.proj(x.reshape(B * N, T, F)))
        z = self.temporal(z)[:, -1].reshape(B, N, self.hidden)
        z = self.spatial(z)
        return torch.relu(self.norm(z))
