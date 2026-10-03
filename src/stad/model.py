"""Сборка «энкодер + голова» и логика деления окна.

Один класс на всю сетку. Любая конфигурация бенчмарка — это пара имён
``(encoder, head)`` плюс размер скрытого слоя, подобранный под общий
бюджет параметров. Благодаря этому сравнение архитектур не смешано со
сравнением ёмкостей — главный дефект большинства таблиц в литературе.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .encoders import Encoder, build_encoder
from .heads import Head, build_head


class Detector(nn.Module):
    """Энкодер + голова как единая обучаемая модель.

    Деление окна. Головы на остатке прогноза (``splits_window=True``)
    требуют, чтобы энкодер **не видел** целевые шаги. Поэтому окно
    режется здесь, в одном месте, а не внутри каждой головы: иначе
    утечка цели легко просачивается незаметно и завышает качество.
    """

    def __init__(self, encoder: Encoder, head: Head) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    # ------------------------------------------------------------------ окна
    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (вход энкодера, цель головы)``."""
        k = self.head.forecast_horizon
        if self.head.splits_window and k > 0:
            if x.shape[2] <= k:
                raise ValueError(f"окно {x.shape[2]} слишком короткое для horizon={k}")
            return x[:, :, :-k, :], x[:, :, -k:, :]
        return x, x

    # ------------------------------------------------------------- forward/loss
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_enc, _ = self._split(x)
        return self.encoder(x_enc)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x_enc, x_tgt = self._split(x)
        h = self.encoder(x_enc)
        return self.head.loss(h, x_tgt if self.head.needs_x else None)

    def d_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_enc, _ = self._split(x)
        with torch.no_grad():
            h = self.encoder(x_enc)
        return self.head.d_loss(h)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        x_enc, x_tgt = self._split(x)
        h = self.encoder(x_enc)
        return self.head.score(h, x_tgt if self.head.needs_x else None)

    # ------------------------------------------------------------------ прочее
    def main_parameters(self):
        d_ids = {id(p) for p in self.head.d_parameters()}
        return (p for p in self.parameters() if id(p) not in d_ids)

    def set_graph(self, A: torch.Tensor) -> None:
        if hasattr(self.encoder, "set_graph"):
            self.encoder.set_graph(A)

    @property
    def adversarial(self) -> bool:
        return self.head.adversarial

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def uses_graph(self) -> bool:
        return bool(self.encoder.uses_graph)


def build_detector(
    encoder: str,
    head: str,
    *,
    hidden: int,
    n_features: int,
    n_nodes: int,
    window: int,
    encoder_kwargs: dict | None = None,
    head_kwargs: dict | None = None,
) -> Detector:
    """Собрать детектор по именам энкодера и головы.

    Важная деталь: голова получает **эффективную** длину окна (с учётом
    отрезанного горизонта прогноза), иначе декодер реконструкции
    попытается восстановить шаги, которых энкодер не видел.
    """
    head_kwargs = dict(head_kwargs or {})
    enc_kwargs = dict(encoder_kwargs or {})

    horizon = int(head_kwargs.get("horizon", 3)) if head == "forecast" else 0
    enc_window = window - horizon if horizon else window
    head_window = horizon if horizon else window

    enc = build_encoder(
        encoder,
        n_features=n_features,
        hidden=hidden,
        n_nodes=n_nodes,
        window=enc_window,
        **enc_kwargs,
    )
    hd = build_head(
        head,
        hidden=hidden,
        n_features=n_features,
        window=head_window,
        n_nodes=n_nodes,
        **head_kwargs,
    )
    return Detector(enc, hd)
