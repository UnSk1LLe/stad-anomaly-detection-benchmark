"""Контракт данных для всех моделей бенчмарка.

Единый формат на входе гарантирует, что архитектуры сравниваются на
идентичных окнах, сплитах и метках — иначе таблица сравнения невалидна
(см. docs/PROTOCOL.md, раздел «Почему протокол важнее архитектуры»).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    """Готовые к обучению окна одного датасета.

    Оси массивов окон: ``[n_windows, N_nodes, T_steps, F_features]``.

    Attributes
    ----------
    X_train, X_val:
        Окна для обучения и валидации. Содержат **только нормальный трафик**:
        все unsupervised-детекторы учат распределение нормы. Окна, пересекающиеся
        с размеченным событием (плюс буфер), исключены — см. ``splits.py``.
    X_test:
        Тестовые окна: норма и аномалии вместе, в исходном временном порядке.
    y_test:
        Поточечные метки ``[n_test, N]`` для конца окна. 1 = узел находится
        в пространственно-временном окне события.
    event_id_test:
        ``[n_test, N]`` идентификатор события или -1. Нужен для event-level
        метрик: одно событие = одна единица учёта, а не сотни точек.
    t_test:
        ``[n_test]`` время конца окна в минутах от начала теста.
    events:
        Реестр событий: ``event_id``, ``t_report`` (минуты, как ``t_test``),
        ``node_lo``/``node_hi`` — диапазон затронутых узлов, ``kind``.
    A:
        ``[N, N]`` матрица смежности (нормализованная). Для неграфовых
        энкодеров игнорируется, но хранится здесь, чтобы все модели
        получали один и тот же объект.
    feature_names:
        Имена F признаков в порядке последней оси.
    meta:
        Произвольные метаданные датасета (шаг дискретизации, источник, версия).
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    event_id_test: np.ndarray
    t_test: np.ndarray
    events: pd.DataFrame
    A: np.ndarray
    feature_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    #: Аномальные окна обучающих дней. Для unsupervised-сетки они не нужны
    #: и исключены из ``X_train`` — модель учит норму. Supervised-арм
    #: (аугментация редкого класса) обучается на обоих классах, поэтому
    #: загрузчик сохраняет их здесь, а не выбрасывает.
    X_train_anomalous: np.ndarray | None = None
    X_val_anomalous: np.ndarray | None = None

    #: ``[n_val]`` время конца каждого валидационного окна, минуты (та же шкала,
    #: что ``t_test``). Нужно, чтобы порог калибровался на валидации: окна там
    #: не сплошные (вокруг событий вырезаны), и подтверждение тревоги не должно
    #: пересекать разрывы. ``None`` — калибровка порога невозможна.
    t_val: np.ndarray | None = None

    #: КАЛИБРОВОЧНАЯ ВЫБОРКА порога: окна дней валидации, где нет размеченных
    #: окон оцениваемых событий (и окон событий другого типа). Буферы вокруг
    #: событий здесь НЕ вырезаются: тестовые «нормальные» окна их содержат, и
    #: порог по чистой норме давал бы на тесте в разы больше тревог, чем бюджет.
    #: Меток теста тут нет. ``X_val`` остаётся чистым и служит ранней остановке.
    X_calib: np.ndarray | None = None
    t_calib: np.ndarray | None = None

    # ------------------------------------------------------------------ utils
    @property
    def n_nodes(self) -> int:
        return self.X_train.shape[1]

    @property
    def window(self) -> int:
        return self.X_train.shape[2]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[3]

    @property
    def prevalence(self) -> float:
        """Базовая частота аномалий — пол для PR-метрик.

        Случайный детектор даёт average precision примерно равный этому
        числу (Lyu, 2026). Любая заявка на качество сравнивается с ним.
        """
        return float(self.y_test.mean())

    def validate(self) -> None:
        """Жёсткие проверки контракта. Вызывается после каждой загрузки."""
        for name in ("X_train", "X_val", "X_test"):
            arr = getattr(self, name)
            if arr.ndim != 4:
                raise ValueError(f"{name}: ожидается 4D [n,N,T,F], получено {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: содержит NaN/inf — предобработка сломана")

        n_test = self.X_test.shape[0]
        if self.y_test.shape != (n_test, self.n_nodes):
            raise ValueError(f"y_test: ожидается {(n_test, self.n_nodes)}, получено {self.y_test.shape}")
        if self.event_id_test.shape != self.y_test.shape:
            raise ValueError("event_id_test и y_test должны совпадать по форме")
        if self.t_test.shape != (n_test,):
            raise ValueError(f"t_test: ожидается ({n_test},), получено {self.t_test.shape}")
        if self.t_val is not None and self.t_val.shape != (self.X_val.shape[0],):
            raise ValueError(f"t_val: ожидается ({self.X_val.shape[0]},), получено {self.t_val.shape}")
        if (self.X_calib is None) != (self.t_calib is None):
            raise ValueError("X_calib и t_calib задаются вместе")
        if self.X_calib is not None:
            if self.X_calib.ndim != 4 or self.X_calib.shape[1:] != self.X_train.shape[1:]:
                raise ValueError(f"X_calib: форма {self.X_calib.shape} несовместима с X_train")
            if self.t_calib.shape != (self.X_calib.shape[0],):
                raise ValueError(f"t_calib: ожидается ({self.X_calib.shape[0]},), получено {self.t_calib.shape}")
        if self.A.shape != (self.n_nodes, self.n_nodes):
            raise ValueError(f"A: ожидается {(self.n_nodes, self.n_nodes)}")
        if len(self.feature_names) != self.n_features:
            raise ValueError("feature_names не соответствует числу признаков")

        # метки и реестр событий должны быть согласованы
        ids = set(np.unique(self.event_id_test[self.event_id_test >= 0]).tolist())
        registry = set(self.events["event_id"].tolist())
        if not ids <= registry:
            raise ValueError(f"event_id_test содержит id вне реестра: {sorted(ids - registry)}")
        if (self.y_test == 1).sum() == 0:
            raise ValueError("в тесте нет ни одной положительной метки — метрики не определены")
        # аномалии должны быть редкими, иначе это не задача детекции
        if self.prevalence > 0.5:
            raise ValueError(f"prevalence={self.prevalence:.3f} > 0.5 — метки, вероятно, инвертированы")
