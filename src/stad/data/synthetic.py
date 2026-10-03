"""Синтетический коридор: smoke-режим и контролируемые эксперименты.

Зачем он нужен, помимо тестов. Реальные метки FT-AED редки (десятки
событий), поэтому дисперсия метрик велика, а тип аномалии не варьируется.
Синтетика даёт две вещи, которых нет в реальных данных:

* **generator-independent инъекция.** Аномалии вводятся физической
  моделью (ударная волна в LWR), а не нейросетью. Это закрывает пробел
  «циркулярность оценки»: если генерировать аномалии GAN и детектировать
  GAN, детектор учится узнавать артефакты генератора.
* **управляемые rarity × amplitude.** Факторный разбор чувствительности,
  который на реальных метках невозможен.

Синтетика **не заменяет** FT-AED в итоговой таблице. Она нужна, чтобы
отладить пайплайн и объяснить, *почему* архитектуры ранжируются так, а не
иначе. Итоговый вывод для диссертации делается на реальных метках.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import align_incidents, windows_from_panel
from .splits import build_split
from .types import SplitData

FEATURES = ["speed", "occupancy", "volume"]


def _corridor_adjacency(n_nodes: int, lanes: int) -> np.ndarray:
    """Смежность коридора: соседние посты × соседние полосы.

    Узлы нумеруются как ``station * lanes + lane``. Связи вдоль движения
    и между смежными полосами одного поста — это та же структура, что в
    FT-AED (49 постов × 4 полосы).
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    stations = n_nodes // lanes
    for s in range(stations):
        for l in range(lanes):
            i = s * lanes + l
            for ds, dl in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                s2, l2 = s + ds, l + dl
                if 0 <= s2 < stations and 0 <= l2 < lanes:
                    A[i, s2 * lanes + l2] = 1.0
    A += np.eye(n_nodes, dtype=np.float32)
    deg = A.sum(1, keepdims=True)
    return (A / deg).astype(np.float32)   # row-normalised


def make_synthetic_corridor(
    *,
    stations: int = 12,
    lanes: int = 3,
    days: int = 10,
    step_min: float = 0.5,
    n_events: int = 24,
    amplitude: float = 0.55,
    event_len_min: tuple[float, float] = (8.0, 35.0),
    window: int = 16,
    seed: int = 0,
    free_flow_kmh: float = 105.0,
) -> SplitData:
    """Сгенерировать коридор с суточным профилем и ударными волнами.

    Parameters
    ----------
    amplitude:
        Относительное падение скорости в эпицентре (0.55 = −55%).
        Главный «ручка» для эксперимента rarity × amplitude.
    n_events:
        Число инцидентов за весь период.
    """
    rng = np.random.default_rng(seed)
    n_nodes = stations * lanes
    steps_per_day = int(24 * 60 / step_min)
    n_t = steps_per_day * days

    ts = pd.date_range("2026-03-02 00:00", periods=n_t, freq=f"{int(step_min)}min"
                       if step_min >= 1 else f"{int(step_min * 60)}s")
    minute_of_day = (np.arange(n_t) * step_min) % (24 * 60)
    weekday = (np.arange(n_t) * step_min // (24 * 60)) % 7

    # суточный профиль: два пика (утро 8:00, вечер 18:00), ночью свободный поток
    def peak(center, width):
        return np.exp(-0.5 * ((minute_of_day - center) / width) ** 2)

    congestion = 0.42 * peak(8 * 60, 70) + 0.34 * peak(18 * 60, 85)
    congestion *= np.where(weekday >= 5, 0.45, 1.0)          # выходные свободнее
    congestion = congestion[:, None]                          # [T,1]

    # пространственный профиль: узкое место ближе к концу коридора
    mileposts = np.repeat(np.linspace(0.0, stations * 0.5, stations), lanes)
    bottleneck = np.exp(-0.5 * ((mileposts - mileposts[int(0.72 * n_nodes)]) / 1.1) ** 2)[None, :]
    lane_bias = np.tile(np.linspace(1.08, 0.9, lanes), stations)[None, :]   # левая полоса быстрее

    speed = free_flow_kmh * lane_bias * (1.0 - congestion * (0.45 + 0.55 * bottleneck))
    speed += rng.normal(0.0, 2.2, size=(n_t, n_nodes))

    # ---------------------------------------------------------------- события
    rows = []
    min_gap = int(120 / step_min)
    candidates = rng.permutation(np.arange(window + min_gap, n_t - min_gap))
    chosen: list[int] = []
    for c in candidates:
        if len(chosen) >= n_events:
            break
        if all(abs(c - p) > min_gap for p in chosen):
            chosen.append(int(c))
    chosen.sort()

    for t0 in chosen:
        dur = float(rng.uniform(*event_len_min))
        n_steps = max(2, int(dur / step_min))
        station = int(rng.integers(1, stations - 1))
        mp = mileposts[station * lanes]
        amp = float(amplitude * rng.uniform(0.7, 1.3))
        kind = str(rng.choice(["crash", "lane_closure", "stalled_vehicle"], p=[0.45, 0.3, 0.25]))

        for k in range(n_steps):
            # ударная волна распространяется ВВЕРХ по потоку (убывающий milepost)
            shock_mp = mp - 0.45 * (k * step_min) / 6.0
            dist = mileposts - shock_mp
            spatial = np.exp(-0.5 * (np.clip(dist, None, 0.0) / 0.9) ** 2)
            spatial *= np.where(dist > 0.12, 0.12, 1.0)      # ниже по потоку эффект слабый
            ramp = min(1.0, (k + 1) / max(1.0, n_steps * 0.25))
            decay = 1.0 if k < n_steps * 0.7 else 0.5
            speed[t0 + k] *= 1.0 - amp * ramp * decay * spatial

        rows.append(
            {
                "start": ts[t0],
                "end": ts[min(n_t - 1, t0 + n_steps)],
                "milepost": float(mp),
                "kind": kind,
                "amplitude": amp,
            }
        )

    speed = np.clip(speed, 4.0, free_flow_kmh * 1.15)

    # занятость и поток из фундаментальной диаграммы (Greenshields)
    v_f, rho_jam = free_flow_kmh * 1.1, 180.0
    rho = np.clip(rho_jam * (1.0 - speed / v_f), 1.0, rho_jam)
    occupancy = np.clip(rho / rho_jam + rng.normal(0, 0.012, speed.shape), 0.0, 1.0)
    volume = np.clip(rho * speed / 60.0 + rng.normal(0, 1.1, speed.shape), 0.0, None)

    panel = pd.DataFrame(
        {
            "ts": np.repeat(ts.to_numpy(), n_nodes),
            "node": np.tile(np.arange(n_nodes), n_t),
            "milepost": np.tile(mileposts, n_t),
            "speed": speed.ravel(),
            "occupancy": occupancy.ravel(),
            "volume": volume.ravel(),
        }
    )
    incidents = pd.DataFrame(rows)

    panel, events = align_incidents(panel, incidents, lead_min=5.0, trail_min=8.0)
    X, y, eid, t_end = windows_from_panel(panel, feature_cols=FEATURES, window=window, stride=1)

    step = float(step_min)
    t_report = ((events["t_report"] - ts[0]) / pd.Timedelta(minutes=1)).to_numpy(dtype=float)
    events = events.assign(t_report=t_report)  # t_end уже абсолютен в минутах от ts[0]

    return build_split(
        X, y, eid, t_end, events,
        A=_corridor_adjacency(n_nodes, lanes),
        feature_names=FEATURES,
        meta={
            "dataset": "synthetic-corridor",
            "stations": stations,
            "lanes": lanes,
            "step_min": step_min,
            "days": days,
            "amplitude": amplitude,
            "seed": seed,
            "mileposts": mileposts.tolist(),
            "generator": "LWR-shockwave (generator-independent)",
        },
    )
