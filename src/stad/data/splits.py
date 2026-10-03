"""Leakage-safe временные сплиты и нормализация.

Два правила, нарушение которых обнуляет результат:

1. **Сплит только по времени, никогда случайный.** Соседние окна
   перекрываются; случайное разбиение помещает почти идентичные окна
   в train и test, и любая модель покажет отличное качество.

2. **Нормализация считается на train и применяется к val/test.**
   Иначе статистики теста утекают в обучение.

Плюс требование unsupervised-постановки: из train и val удаляются окна,
пересекающиеся с размеченным событием (с буфером). Модель учит норму.
"""
from __future__ import annotations

import numpy as np

from .types import SplitData


def time_split_indices(n: int, *, train: float = 0.6, val: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Последовательные индексы train/val/test без перемешивания."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError(f"некорректные доли: train={train}, val={val}")
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return np.arange(0, i_tr), np.arange(i_tr, i_va), np.arange(i_va, n)


def drop_anomalous_windows(
    X: np.ndarray, y: np.ndarray, *, buffer: int = 6
) -> np.ndarray:
    """Индексы окон, в которых нет ни одного положительного узла (с буфером).

    ``buffer`` — сколько окон до и после каждого «грязного» окна тоже
    выбросить: событие проявляется в данных раньше отчёта, и остаточное
    влияние тянется после.
    """
    dirty = y.any(axis=1)
    if buffer > 0:
        pad = dirty.copy()
        for k in range(1, buffer + 1):
            pad[k:] |= dirty[:-k]
            pad[:-k] |= dirty[k:]
        dirty = pad
    return np.where(~dirty)[0]


class Standardizer:
    """Пер-признаковая z-нормализация, обученная на train.

    Статистики считаются по осям ``(n, N, T)`` — то есть один scale на
    признак, общий для всех узлов. Пер-узловая нормализация убрала бы
    пространственную гетерогенность, которую графовые модели как раз
    должны использовать (ср. вывод Dong et al., 2026 о per-sensor
    различиях как источнике архитектурно-специфичных отказов).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer не обучен")
        return ((X - self.mean_) / self.std_).astype(np.float32)


def build_split(
    X: np.ndarray,
    y: np.ndarray,
    eid: np.ndarray,
    t_end: np.ndarray,
    events,
    A: np.ndarray,
    feature_names: list[str],
    *,
    train: float = 0.6,
    val: float = 0.15,
    clean_buffer: int = 6,
    meta: dict | None = None,
) -> SplitData:
    """Собрать :class:`SplitData` из нарезанных окон."""
    idx_tr, idx_va, idx_te = time_split_indices(len(X), train=train, val=val)

    keep_tr = idx_tr[np.isin(idx_tr, idx_tr[drop_anomalous_windows(X[idx_tr], y[idx_tr], buffer=clean_buffer)])]
    keep_va = idx_va[drop_anomalous_windows(X[idx_va], y[idx_va], buffer=clean_buffer)]
    if len(keep_tr) < 32:
        raise ValueError(
            f"после очистки в train осталось {len(keep_tr)} окон — "
            "уменьшите clean_buffer или возьмите больше данных"
        )

    scaler = Standardizer().fit(X[keep_tr])

    # событийные времена — в ту же шкалу минут, что t_end
    ev = events.copy()
    if "t_report" in ev.columns and np.issubdtype(ev["t_report"].dtype, np.datetime64):
        t0 = ev["t_report"].min()
        raise ValueError(
            "events.t_report должен быть в минутах той же шкалы, что t_end; "
            "сконвертируйте до вызова build_split"
        )

    # в реестре оставляем только события, присутствующие в тесте
    ids_test = set(np.unique(eid[idx_te][eid[idx_te] >= 0]).tolist())
    ev = ev[ev["event_id"].isin(ids_test)].reset_index(drop=True)

    data = SplitData(
        X_train=scaler.transform(X[keep_tr]),
        X_val=scaler.transform(X[keep_va]) if len(keep_va) else scaler.transform(X[keep_tr][:8]),
        X_test=scaler.transform(X[idx_te]),
        y_test=y[idx_te].astype(np.int64),
        event_id_test=eid[idx_te].astype(np.int64),
        t_test=t_end[idx_te].astype(np.float64),
        events=ev,
        A=A.astype(np.float32),
        feature_names=list(feature_names),
        t_val=t_end[keep_va].astype(np.float64) if len(keep_va) else None,
        meta={
            **(meta or {}),
            "n_train_windows": int(len(keep_tr)),
            "n_val_windows": int(len(keep_va)),
            "n_test_windows": int(len(idx_te)),
            "n_events_test": int(len(ev)),
            "dropped_train_windows": int(len(idx_tr) - len(keep_tr)),
            "scaler_mean": scaler.mean_.ravel().tolist(),
            "scaler_std": scaler.std_.ravel().tolist(),
        },
    )
    data.validate()
    return data
