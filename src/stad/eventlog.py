"""Пособытийная таблица клетки: какое событие найдено, когда и с каким кредитом.

``padf`` клетки — среднее кредита по событиям; эта таблица хранит сами слагаемые.
Она нужна статистике, где единица — событие, а не блок «фолд × сид» (кластерный
bootstrap по событиям), и диагностике прогона. Считается теми же функциями, что и
``runs.csv``: порог — уже откалиброванный на валидации (``row["threshold"]``),
свёртка по узлам и подтверждение тревоги — те же, кредит события — ``padf`` от
его задержки. Поэтому ``credit.mean()`` совпадает с ``padf`` клетки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.types import SplitData
from .metrics import eval_segments, evaluation_view, event_delays, padf

#: Колонки таблицы (без идентификаторов клетки ``config, dataset, seed``).
EVENT_COLUMNS: tuple[str, ...] = ("event_id", "t_report", "detected", "delay_min", "credit")


def event_table(
    scores: np.ndarray,
    data: SplitData,
    threshold: float,
    *,
    half_life_min: float,
    persistence: int,
    node_reduce: str = "max",
) -> pd.DataFrame:
    """Строка на каждое событие теста: ``event_id, t_report, detected, delay_min, credit``.

    ``scores`` — сырые score теста ``[n_test, N]``, как их сохраняет раннер;
    свёртка к гранулярности меток выполняется здесь (:func:`evaluation_view`).
    ``delay_min`` у необнаруженного события — NaN, кредит — 0.
    """
    s, _, eid = evaluation_view(scores, data, reduce=node_reduce)
    det = event_delays(s, eid, data.t_test, data.events, threshold,
                       persistence=persistence, segments=eval_segments(data))
    ev_ids = data.events["event_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(det["event_id"].to_numpy(dtype=np.int64), ev_ids):
        raise RuntimeError("порядок событий в event_delays разошёлся с реестром событий")
    delays = det["delay_min"].to_numpy(dtype=float)
    credit = np.array([padf(np.array([d]), half_life_min=half_life_min) for d in delays], dtype=float)
    return pd.DataFrame({
        "event_id": ev_ids,
        "t_report": data.events["t_report"].to_numpy(dtype=float),
        "detected": det["detected"].to_numpy(dtype=bool),
        "delay_min": np.where(np.isfinite(delays), delays, np.nan),
        "credit": credit,
    })


__all__ = ["EVENT_COLUMNS", "event_table"]
