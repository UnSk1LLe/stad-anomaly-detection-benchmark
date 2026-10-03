"""Графовые энкодеры: фиксированный граф и обучаемое внимание.

Две архитектуры различаются **ровно одним**: откуда берётся
пространственная структура.

* :class:`GCNGRU` — фиксированная физическая смежность коридора. Это
  семейство, которое бенчмарк FT-AED (Coursey et al., 2024) назвал
  наиболее перспективным, и там же показано, что отказ от
  пространственных связей ухудшает качество. Референсная точка.
* :class:`GATLSTM` — обучаемое внимание по маске графа. В работе
  «Toward explainable automatic incident detection» (2025) комбинация
  GAT + LSTM оказалась лучшей из перебранных пар «пространственный ×
  временной модуль»; независимое подтверждение на соседнем домене —
  Veesam et al. (2025).

Гипотеза, которую проверяет пара: аномалия есть **разрыв**
пространственной корреляции, поэтому фиксированное сглаживание по графу
может размывать именно тот сигнал, который нужно поймать, а обучаемое
внимание — нет.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import Encoder


class GCNGRU(Encoder):
    """GRU по времени + диффузия по фиксированному графу (DCRNN-подобно).

    Порядок «время, затем пространство» выбран сознательно: он сохраняет
    полную временную динамику до смешивания узлов. Обратный порядок
    (сначала графовая свёртка покадрово) сглаживает локальный всплеск
    раньше, чем его увидит рекуррентная часть.
    """

    uses_graph = True

    def __init__(self, n_features: int, hidden: int, n_nodes: int, window: int, *, diffusion_steps: int = 2) -> None:
        super().__init__(n_features, hidden, n_nodes, window)
        self.K = diffusion_steps
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.diffusion = nn.Linear(hidden * (diffusion_steps + 1), hidden)
        self.norm = nn.LayerNorm(hidden)
        self.register_buffer("A", torch.eye(n_nodes), persistent=False)

    def set_graph(self, A: torch.Tensor) -> None:
        self.A = A.to(self.A.device if self.A is not None else A.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check(x)
        B, N, T, F = x.shape
        h, _ = self.gru(x.reshape(B * N, T, F))
        h = h[:, -1].reshape(B, N, self.hidden)

        # диффузия: [h, Ah, A²h, ...] — многошаговое распространение по коридору
        hops = [h]
        cur = h
        for _ in range(self.K):
            cur = torch.einsum("ij,bjh->bih", self.A, cur)
            hops.append(cur)
        out = self.diffusion(torch.cat(hops, dim=-1))
        return torch.relu(self.norm(out))


class GATLSTM(Encoder):
    """LSTM по времени + одноголовое graph attention по маске смежности.

    Маска графа сохранена: внимание учится **весам** связей, но не
    создаёт связи между физически несвязанными участками коридора. Это
    отличает архитектуру от полного self-attention по узлам (который
    реализован в :class:`~stad.encoders.sequence.NodeTransformer`) и
    делает сравнение «фиксированные веса против обучаемых» чистым.
    """

    uses_graph = True

    def __init__(self, n_features: int, hidden: int, n_nodes: int, window: int, *, leaky: float = 0.2) -> None:
        super().__init__(n_features, hidden, n_nodes, window)
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.att_src = nn.Linear(hidden, 1, bias=False)
        self.att_dst = nn.Linear(hidden, 1, bias=False)
        self.value = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.leaky = nn.LeakyReLU(leaky)
        self.register_buffer("mask", torch.eye(n_nodes, dtype=torch.bool), persistent=False)

    def set_graph(self, A: torch.Tensor) -> None:
        self.mask = (A > 0).to(A.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check(x)
        B, N, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N, T, F))
        h = h[:, -1].reshape(B, N, self.hidden)

        # аддитивное внимание GAT: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j)
        e = self.leaky(self.att_src(h) + self.att_dst(h).transpose(1, 2))   # [B,N,N]
        e = e.masked_fill(~self.mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum("bij,bjh->bih", alpha, self.value(h))
        return torch.relu(self.norm(out))


class DirectedHypergraph(Encoder):
    """Направленный гиперграф + Bi-GLU (по мотивам DHMPN).

    Отличие от GCN: сообщение проходит не по парным связям, а через
    **гиперребра** — группы узлов (в коридоре это естественно: все полосы
    одного поста, скользящая группа соседних постов). Узлы агрегируются
    в гиперребро, гиперребро — обратно в узлы. Направленность задаётся
    разделением на входящий и исходящий поток.

    В статье DHMPN такая структура позиционируется как дешёвая
    альтернатива трансформерам, пригодная для edge-устройств, и
    заявляется превосходство над SOTA графовыми моделями. Здесь она
    проверяется при выровненном бюджете параметров.
    """

    uses_graph = True

    def __init__(
        self,
        n_features: int,
        hidden: int,
        n_nodes: int,
        window: int,
        *,
        group: int = 3,
        n_lanes: int = 3,
    ) -> None:
        super().__init__(n_features, hidden, n_nodes, window)
        self.temporal = nn.GRU(n_features, hidden, batch_first=True)
        self.to_edge_fwd = nn.Linear(hidden, hidden)
        self.to_edge_bwd = nn.Linear(hidden, hidden)
        self.glu = nn.Linear(hidden * 2, hidden * 2)      # Bi-GLU: value ⊗ gate
        self.to_node = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.register_buffer("incidence", self._build_incidence(n_nodes, group, n_lanes), persistent=False)

    @staticmethod
    def _build_incidence(n_nodes: int, group: int, n_lanes: int) -> torch.Tensor:
        """Матрица инцидентности ``[n_edges, n_nodes]``, row-normalised.

        Два типа гиперребер: (1) все полосы одного поста — «поперечное
        сечение»; (2) скользящее окно из ``group`` постов — «участок».
        """
        edges: list[list[int]] = []
        stations = max(1, n_nodes // n_lanes)
        for s in range(stations):
            edges.append([s * n_lanes + l for l in range(n_lanes) if s * n_lanes + l < n_nodes])
        for s in range(stations - group + 1):
            edges.append(
                [s2 * n_lanes + l for s2 in range(s, s + group) for l in range(n_lanes) if s2 * n_lanes + l < n_nodes]
            )
        H = torch.zeros(len(edges), n_nodes)
        for i, members in enumerate(edges):
            if members:
                H[i, members] = 1.0 / len(members)
        return H

    def set_graph(self, A: torch.Tensor) -> None:  # noqa: ARG002 - гиперграф задан инцидентностью
        """Гиперструктура определяется геометрией коридора, не матрицей A."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.check(x)
        B, N, T, F = x.shape
        h, _ = self.temporal(x.reshape(B * N, T, F))
        h = h[:, -1].reshape(B, N, self.hidden)

        H = self.incidence                                     # [E, N]
        e_fwd = torch.einsum("en,bnh->beh", H, self.to_edge_fwd(h))
        e_bwd = torch.einsum("en,bnh->beh", H, self.to_edge_bwd(h))

        gated = self.glu(torch.cat([e_fwd, e_bwd], dim=-1))
        value, gate = gated.chunk(2, dim=-1)
        e = value * torch.sigmoid(gate)                        # Bi-GLU

        back = torch.einsum("en,beh->bnh", H, e) / (H.sum(0).clamp(min=1e-6)[None, :, None])
        return torch.relu(self.norm(h + self.to_node(back)))
