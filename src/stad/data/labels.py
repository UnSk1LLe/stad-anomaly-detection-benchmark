"""Выравнивание журнала инцидентов с показаниями детекторов.

Это самый недооценённый шаг эксперимента. От него зависит, что именно
модель считает аномалией, и он задаёт потолок всех метрик. Три решения,
которые здесь зафиксированы явно, а не спрятаны в предобработке:

1. **Пространственный радиус.** Инцидент влияет на детекторы выше по
   потоку, а не ниже: затор распространяется против движения. Поэтому окно
   по пространству асимметрично (``upstream_mi`` >> ``downstream_mi``).
   Эмпирическое обоснование асимметрии — Detection Rate of Congestion
   Patterns Comparing Multiple Traffic Sensor Technologies (2024): Bluetooth
   почти непригоден для коротких инцидентов именно потому, что измеряет
   вниз по потоку.

2. **Опережающее окно (`lead_min`).** Время официального отчёта — не время
   события. Сигнал в данных появляется раньше. Если считать всё до отчёта
   нормой, детектор штрафуется за раннее обнаружение, то есть ровно за то,
   что нам нужно. Бенчмарк FT-AED (Coursey et al., 2024) измеряет сокращение
   задержки относительно отчёта, а CCB-GraphGAN (Nouri et al., 2026)
   показывает −5 мин при FPR 1% — значит сигнал там есть.

3. **Буфер (`buffer_min`).** Окна вокруг события исключаются из обучающей
   выборки, чтобы unsupervised-модель не учила аномалию как норму.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def align_incidents(
    flow: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    upstream_mi: float = 1.5,
    downstream_mi: float = 0.3,
    lead_min: float = 10.0,
    trail_min: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Присвоить поточечные метки и построить реестр событий.

    Parameters
    ----------
    flow:
        Столбцы ``ts`` (datetime), ``node`` (int), ``milepost`` (float)
        и произвольные признаки.
    incidents:
        Столбцы ``start``, ``end`` (datetime), ``milepost`` (float),
        необязательно ``kind`` (str).
    upstream_mi, downstream_mi:
        Радиус влияния вверх и вниз по потоку, в милях. Направление
        определяется как возрастание ``milepost`` = направление движения.
    lead_min:
        За сколько минут до ``start`` начинать считать точки положительными.
    trail_min:
        Сколько минут после ``end`` ещё считать положительными (разгрузка).

    Returns
    -------
    flow_labelled:
        ``flow`` с добавленными ``y_point`` (0/1) и ``event_id`` (-1 = нет).
    events:
        Реестр: ``event_id``, ``t_report`` (= ``start``), ``start``, ``end``,
        ``milepost``, ``kind``, ``n_points`` (число помеченных точек).
    """
    required_flow = {"ts", "node", "milepost"}
    if not required_flow <= set(flow.columns):
        raise ValueError(f"flow: нужны столбцы {sorted(required_flow)}")
    required_inc = {"start", "end", "milepost"}
    if not required_inc <= set(incidents.columns):
        raise ValueError(f"incidents: нужны столбцы {sorted(required_inc)}")

    out = flow.sort_values("ts").reset_index(drop=True).copy()
    out["y_point"] = 0
    out["event_id"] = -1

    inc = incidents.reset_index(drop=True).copy()
    inc["event_id"] = np.arange(len(inc))
    if "kind" not in inc.columns:
        inc["kind"] = "unknown"

    n_points = []
    for row in inc.itertuples():
        t0 = row.start - pd.Timedelta(minutes=lead_min)
        t1 = row.end + pd.Timedelta(minutes=trail_min)
        # асимметрия: вверх по потоку = меньшие milepost в направлении движения
        lo = row.milepost - upstream_mi
        hi = row.milepost + downstream_mi
        mask = out["ts"].between(t0, t1) & out["milepost"].between(lo, hi)
        n = int(mask.sum())
        n_points.append(n)
        # при пересечении событий побеждает более раннее (детерминированно)
        fresh = mask & (out["event_id"] < 0)
        out.loc[mask, "y_point"] = 1
        out.loc[fresh, "event_id"] = row.event_id

    inc["n_points"] = n_points
    inc["t_report"] = inc["start"]

    orphans = inc.loc[inc["n_points"] == 0, "event_id"].tolist()
    if orphans:
        # событие вне покрытия детекторов нельзя ни обнаружить, ни пропустить:
        # оставлять его в реестре значит занижать recall по построению
        inc = inc[inc["n_points"] > 0].reset_index(drop=True)

    cols = ["event_id", "t_report", "start", "end", "milepost", "kind", "n_points"]
    return out, inc[cols]


def windows_from_panel(
    flow: pd.DataFrame,
    *,
    feature_cols: list[str],
    window: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Нарезать панель ``(ts × node)`` на окна.

    Метка окна берётся с **последнего** шага: детектор должен принимать
    решение в реальном времени, не подглядывая в будущее.

    Returns
    -------
    X : ``[n, N, T, F]``
    y : ``[n, N]``
    eid : ``[n, N]``
    t_end : ``[n]`` минуты от первого отсчёта
    """
    flow = flow.sort_values(["ts", "node"])
    ts = np.sort(flow["ts"].unique())
    nodes = np.sort(flow["node"].unique())
    n_t, n_n = len(ts), len(nodes)

    t_index = {v: i for i, v in enumerate(ts)}
    n_index = {v: i for i, v in enumerate(nodes)}

    feat = np.full((n_t, n_n, len(feature_cols)), np.nan, dtype=np.float32)
    y = np.zeros((n_t, n_n), dtype=np.int64)
    eid = np.full((n_t, n_n), -1, dtype=np.int64)

    ti = flow["ts"].map(t_index).to_numpy()
    ni = flow["node"].map(n_index).to_numpy()
    feat[ti, ni] = flow[feature_cols].to_numpy(dtype=np.float32)
    if "y_point" in flow.columns:
        y[ti, ni] = flow["y_point"].to_numpy()
    if "event_id" in flow.columns:
        eid[ti, ni] = flow["event_id"].to_numpy()

    # пропуски в сенсорных данных — норма; заполняем вперёд, затем назад
    for f in range(feat.shape[-1]):
        col = feat[:, :, f]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(n_t)[:, None], 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            col = col[idx, np.arange(n_n)[None, :]]
            col = np.nan_to_num(col, nan=np.nanmedian(feat[:, :, f]) if np.isfinite(np.nanmedian(feat[:, :, f])) else 0.0)
            feat[:, :, f] = col

    starts = np.arange(0, n_t - window + 1, stride)
    X = np.stack([feat[s : s + window] for s in starts])          # [n, T, N, F]
    X = np.transpose(X, (0, 2, 1, 3)).astype(np.float32)          # [n, N, T, F]
    ends = starts + window - 1
    step_min = float(np.median(np.diff(ts)) / np.timedelta64(1, "m")) if n_t > 1 else 1.0
    return X, y[ends], eid[ends], ends.astype(np.float64) * step_min
