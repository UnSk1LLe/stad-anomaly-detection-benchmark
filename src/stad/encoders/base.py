"""Интерфейс пространственно-временного энкодера.

Контракт один для всех: ``[B, N, T, F] -> [B, N, H]``. Энкодер не знает
ничего об аномалиях — он только строит представление узла. Решение
«аномалия или нет» принимает голова (``stad.heads``).

Такое разделение и есть предмет эксперимента. ST-GNN — это энкодер, а не
детектор; в литературе эти две оси смешаны, из-за чего невозможно понять,
чем вызван выигрыш. Разделив их, мы можем измерить вклад каждой.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    """Базовый энкодер.

    Attributes
    ----------
    uses_graph:
        Использует ли энкодер матрицу смежности. Нужно для фигуры
        «вклад пространственного prior»: сравнение графовых и неграфовых
        при выровненном бюджете параметров.
    """

    uses_graph: bool = False

    def __init__(self, n_features: int, hidden: int, n_nodes: int, window: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.n_nodes = n_nodes
        self.window = window

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: [B, N, T, F] -> [B, N, H]``."""

    # ------------------------------------------------------------------ utils
    def check(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"{type(self).__name__}: ожидается [B,N,T,F], получено {tuple(x.shape)}")
        if x.shape[1] != self.n_nodes:
            raise ValueError(f"{type(self).__name__}: N={x.shape[1]}, ожидалось {self.n_nodes}")
        if x.shape[-1] != self.n_features:
            raise ValueError(f"{type(self).__name__}: F={x.shape[-1]}, ожидалось {self.n_features}")

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
