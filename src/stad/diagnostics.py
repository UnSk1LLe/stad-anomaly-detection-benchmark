"""Диагностика протокола готового прогона: по сохранённым score, без обучения.

Зачем (docs/TZ_PROTOCOL_V2.md, задача 1.1). Числа ``runs.csv`` правдоподобны, но
ничего не говорят о том, на чём держатся: сколько тревог реально выдаёт порог,
где лежит случайный контроль, что даёт score без модели, сколько независимых
наблюдений за средним. Этот модуль считает это теми же функциями, что и раннер
(``stad.metrics``), чтобы диагностика не могла разойтись с таблицей.

Пункты (буквы — разделы ``DIAGNOSTICS.md``):

A. **Единица бюджета тревог.** Ячейка ложной тревоги — то, что считает
   ``alarms_per_hour``; окно тревоги — окно с хотя бы одной такой ячейкой;
   **эпизод** — начало непрерывной серии окон тревоги внутри сегмента. Склейки
   серий через разрыв здесь нет: это изменение метрики (задача 2.1, класс B).
B. **Нулевое распределение случайного score** по многим сидам и полоса среднего
   прогона — в двух вариантах числа независимых значений на фолд.
C. **ORACLE: равная реальная частота тревог.** Порог по нормальным окнам теста —
   только диагностика, в ранжировании не используется никогда.
D. **Безмодельные score**: средняя занятость и минус средняя скорость по сети.
E. **Псевдорепликация сидов**: разброс по сидам внутри фолда.
F. **Парные сравнения с референсом**: блоки, фолды и кластерный bootstrap по
   событиям рядом.

Функции модуля чистые (ничего не печатают); точка входа для скрипта —
:func:`run_diagnostics`. Выводов об архитектурах здесь нет.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines.controls import RandomScorer
from .data.types import SplitData
from .eventlog import event_table
from .experiment import ExperimentConfig
from .metrics import (
    confirm_alarms,
    eval_segments,
    evaluation_view,
    event_level_report,
    full_report,
    make_calibration,
)
from .metrics.stats import paired_bootstrap_test
from .registry import REFERENCE, get_grid
from .report import compare_to_reference
from .runner import run_config
from .train import set_seed

#: Опорные числа docs/TZ_PROTOCOL_V2.md, раздел 1 (пп. 2–5), для пилотного прогона
#: ``reports/ft_aed_cv`` (метки crash). Ключ -> (описание, значение, допуск). Допуск —
#: половина единицы последнего знака, с которым число записано в ТЗ. Нужны только
#: для сверки ``--tz-reference``; в расчётах не участвуют.
TZ_REFERENCE: dict[str, tuple[str, float, float]] = {
    "n_events_test": ("событий теста, сумма по фолдам", 35, 0.0),
    "random_expected_padf": ("ожидаемый padf случайного score (в ТЗ — 200 сидов × 4 фолда)", 0.104, 0.0005),
    "padf_mean.base_iforest": ("padf base_iforest (runs.csv)", 0.281, 0.0005),
    "padf_mean.cand_gcngru_bigan": ("padf cand_gcngru_bigan (runs.csv)", 0.212, 0.0005),
    "padf_mean.base_snd": ("padf base_snd (runs.csv)", 0.210, 0.0005),
    "padf_mean.cand_gcngru_recon": ("padf референса (runs.csv)", 0.147, 0.0005),
    "padf_mean.base_pca": ("padf base_pca (runs.csv)", 0.163, 0.0005),
    "padf_mean.cand_hypergraph_flow": ("padf cand_hypergraph_flow (runs.csv)", 0.167, 0.0005),
    "padf_mean.cand_gatlstm_recon": ("padf cand_gatlstm_recon (runs.csv)", 0.115, 0.0005),
    "padf_mean.deep_transformer_contrastive": ("padf deep_transformer_contrastive (runs.csv)", 0.121, 0.0005),
    "calib_hours_mean": ("часов калибровки на фолд, среднее («~11 ч»)", 11.0, 0.5),
    "calib_budget_windows_min": ("бюджет калибровки в окнах, минимум по фолдам", 2, 0.0),
    "calib_budget_windows_max": ("бюджет калибровки в окнах, максимум по фолдам", 3, 0.0),
    "calib_episodes_min": ("эпизодов тревоги в калибровке, минимум по клеткам", 1, 0.0),
    "calib_episodes_max": ("эпизодов тревоги в калибровке, максимум по клеткам", 2, 0.0),
    # «от 0 до 6.05/ч»: уровень агрегации в ТЗ не указан; число совпадает со средним по сидам
    # конфигурации в фолде (максимум отдельной клетки больше — его печатает отчёт)
    "alarm_windows_per_hour_fold_min": ("окон тревоги/ч, среднее по сидам в фолде, минимум", 0.0, 0.005),
    "alarm_windows_per_hour_fold_max": ("окон тревоги/ч, среднее по сидам в фолде, максимум", 6.05, 0.005),
    "alarm_windows_per_hour_config_min": ("окон тревоги/ч, среднее конфигурации, минимум", 0.28, 0.005),
    "alarm_windows_per_hour_config_max": ("окон тревоги/ч, среднее конфигурации, максимум", 2.89, 0.005),
    "episodes_per_hour_config_min": ("эпизодов/ч, среднее конфигурации, минимум", 0.07, 0.005),
    "episodes_per_hour_config_max": ("эпизодов/ч, среднее конфигурации, максимум", 0.29, 0.005),
    # 0.355 — клетки ctrl_random прогона (его сиды), а не среднее нулевого распределения
    "random_oracle_padf_1": ("ORACLE: padf ctrl_random (сиды прогона) при 1.0 окна/ч", 0.355, 0.0005),
    "model_free.occ_mean.average_precision": ("AP средней занятости (без модели)", 0.351, 0.0005),
    "model_free.occ_mean.roc_auc": ("ROC средней занятости (без модели)", 0.702, 0.0005),
    "ap_mean.cand_gcngru_recon": ("AP референса (runs.csv)", 0.363, 0.0005),
    "roc_mean.cand_gcngru_recon": ("ROC референса (runs.csv)", 0.707, 0.0005),
    "ap_mean.base_pca": ("AP base_pca (runs.csv)", 0.377, 0.0005),
    "roc_mean.base_pca": ("ROC base_pca (runs.csv)", 0.710, 0.0005),
    "ap_mean.ctrl_untrained": ("AP ctrl_untrained (runs.csv)", 0.360, 0.0005),
    "roc_mean.ctrl_untrained": ("ROC ctrl_untrained (runs.csv)", 0.688, 0.0005),
    "block_ci_lo.base_iforest": ("блочный ДИ base_iforest − референс, нижняя граница", 0.073, 0.0005),
    "block_ci_hi.base_iforest": ("блочный ДИ base_iforest − референс, верхняя граница", 0.200, 0.0005),
    "event_ci_lo.base_iforest": ("ДИ по событиям base_iforest − референс, нижняя", 0.024, 0.0005),
    "event_ci_hi.base_iforest": ("ДИ по событиям base_iforest − референс, верхняя", 0.257, 0.0005),
    "event_half_width_min": ("полуширина ДИ по событиям, минимум по парам", 0.10, 0.005),
    "event_half_width_max": ("полуширина ДИ по событиям, максимум по парам", 0.15, 0.005),
}
#: Полоса случайного контроля из ТЗ (определение в ТЗ не указано) и конфигурации
#: с нулевым разбросом по сидам внутри фолда — те же источник и прогон.
TZ_RANDOM_BAND: tuple[float, float] = (0.040, 0.186)
TZ_ZERO_VARIANCE: tuple[str, ...] = ("base_pca", "base_snd", "base_california", "ctrl_untrained")

#: Безмодельные score: имя -> (признак, знак, описание). Последний шаг окна, среднее по сети.
MODEL_FREE_SCORES: dict[str, tuple[str, float, str]] = {
    "occ_mean": ("occupancy", 1.0, "средняя по сети стандартизованная занятость"),
    "neg_speed_mean": ("speed", -1.0, "минус средняя по сети стандартизованная скорость"),
}

_ORACLE_NOTE = ("ORACLE: порог подобран по нормальным окнам ТЕСТА (тестовые метки). Только "
                "диагностика механизма метрики, в ранжировании не используется никогда.")


@dataclass(frozen=True)
class Protocol:
    """Параметры оценки прогона. Берутся из ``manifest.json``: то, с чем прогон считался."""

    alarm_budget_per_hour: float
    half_life_min: float
    persistence: int
    node_reduce: str

    def report_kwargs(self) -> dict:
        return {"alarm_budget_per_hour": self.alarm_budget_per_hour, "half_life_min": self.half_life_min,
                "persistence": self.persistence, "reduce": self.node_reduce}


def protocol_for_run(cfg: ExperimentConfig, manifest: dict) -> tuple[Protocol, list[str]]:
    """Протокол прогона: значения манифеста, при отсутствии — конфига. Расхождения — в примечаниях."""
    notes: list[str] = []
    vals = {}
    for key in ("alarm_budget_per_hour", "half_life_min", "persistence", "node_reduce"):
        own = getattr(cfg, key)
        got = manifest.get(key, own)
        if got != own:
            notes.append(f"`{key}`: в манифесте {got}, в конфиге {own} — взято из манифеста")
        vals[key] = got
    return Protocol(float(vals["alarm_budget_per_hour"]), float(vals["half_life_min"]),
                    int(vals["persistence"]), str(vals["node_reduce"])), notes


#: Что в метаданных датасета должно совпасть с ``manifest.json["datasets"]``: иначе
#: конфиг собрал не те данные, на которых считался прогон (например, метки crash
#: против both при одинаковых формах score).
DATASET_CHECK_KEYS: tuple[str, ...] = ("label_source", "step_min", "window", "n_calib_windows",
                                       "n_test_windows", "n_events_test")


def dataset_mismatches(datasets: dict[str, SplitData], manifest: dict) -> list[str]:
    """Расхождения метаданных собранных датасетов с манифестом прогона (пусто — совпадают)."""
    out = []
    saved = manifest.get("datasets", {})
    for name, data in datasets.items():
        meta = saved.get(name)
        if not isinstance(meta, dict):
            continue
        for key in DATASET_CHECK_KEYS:
            if key not in meta or key not in data.meta:
                continue
            a, b = data.meta[key], meta[key]
            try:
                same = float(a) == float(b)
            except (TypeError, ValueError):
                same = str(a) == str(b)
            if not same:
                out.append(f"{name}.{key}: собрано {a!r}, в манифесте {b!r}")
    return out


# ======================================================== A. единицы тревог
def alarm_episode_starts(alarm_windows: np.ndarray, segments: np.ndarray | None = None) -> np.ndarray:
    """Начала эпизодов: окно тревоги, перед которым в том же сегменте окна тревоги нет.

    ``alarm_windows`` — ``[n]`` (или ``[n, M]``, тогда окно — любая ячейка). Серия не
    переходит границу сегмента (ночной разрыв, вырезанные окна): там начинается новая.
    Склейки серий через короткий разрыв нет — это задача 2.1 (класс B).
    """
    a = np.asarray(alarm_windows, dtype=bool)
    a = a.reshape(len(a), -1).any(axis=1)
    prev = np.zeros_like(a)
    prev[1:] = a[:-1]
    if segments is not None:
        seg = np.asarray(segments)
        same = np.zeros_like(a)
        same[1:] = seg[1:] == seg[:-1]
        prev &= same
    return a & ~prev


def alarm_units(
    s: np.ndarray, y: np.ndarray, threshold: float, *, step_min: float, persistence: int,
    segments: np.ndarray | None,
) -> dict[str, float]:
    """Ячейки, окна и эпизоды подтверждённых ложных тревог по уже свёрнутым score ``[n, M]``.

    Ячейка — ``confirm_alarms(s > thr) & (y == 0)``, ровно числитель ``alarms_per_hour``.
    """
    cells = confirm_alarms(s > threshold, persistence, segments=segments) & (y == 0)
    windows = cells.reshape(len(cells), -1).any(axis=1)
    episodes = alarm_episode_starts(windows, segments)
    hours = s.shape[0] * step_min / 60.0
    n_c, n_w, n_e = int(cells.sum()), int(windows.sum()), int(episodes.sum())

    def per_h(k: int) -> float:
        return k / hours if hours > 0 else float("nan")

    return {"hours": hours, "alarm_cells": n_c, "alarm_windows": n_w, "episodes": n_e,
            "alarm_cells_per_hour": per_h(n_c), "alarm_windows_per_hour": per_h(n_w),
            "episodes_per_hour": per_h(n_e)}


def alarm_units_test(scores: np.ndarray, data: SplitData, threshold: float, *, persistence: int,
                     node_reduce: str) -> dict[str, float]:
    """Единицы тревог на тесте после свёртки по узлам (:func:`evaluation_view`)."""
    s, y, _ = evaluation_view(scores, data, reduce=node_reduce)
    return alarm_units(s, y, threshold, step_min=float(data.meta.get("step_min", 0.5)),
                       persistence=persistence, segments=eval_segments(data))


def alarm_units_calib(calib_scores: np.ndarray, data: SplitData, threshold: float, *, persistence: int,
                      node_reduce: str) -> dict[str, float]:
    """Единицы тревог в калибровочной выборке — то, на что опирается порог."""
    cal = make_calibration(calib_scores, data, reduce=node_reduce)
    return alarm_units(cal.scores, np.zeros(cal.scores.shape, dtype=np.int64), threshold,
                       step_min=cal.step_min, persistence=persistence, segments=cal.segments)


def oracle_report(scores: np.ndarray, data: SplitData, rate: float, protocol: Protocol) -> dict[str, float]:
    """ORACLE: порог на не более ``rate`` подтверждённых ложных ячеек/ч по нормальным окнам ТЕСТА.

    Только диагностика (равная реальная частота тревог): порог видит тестовые метки.
    При корридор-уровневых метках ячейка после свёртки по узлам — окно.
    """
    s, y, eid = evaluation_view(scores, data, reduce=protocol.node_reduce)
    rep = event_level_report(
        s, y, eid, data.t_test, data.events, alarm_budget_per_hour=rate,
        half_life_min=protocol.half_life_min, step_min=float(data.meta.get("step_min", 0.5)),
        persistence=protocol.persistence, calibration=None, segments=eval_segments(data),
    )
    if rep["threshold_source"] != "test_normals_ORACLE":
        raise RuntimeError("оракульный порог должен помечаться как test_normals_ORACLE")
    return {"rate_per_hour": rate, "threshold_ORACLE": rep["threshold"], "padf_ORACLE": rep["padf"],
            "event_recall_ORACLE": rep["event_recall"], "alarms_per_hour_ORACLE": rep["alarms_per_hour"]}


def cell_diagnostics(
    scores: np.ndarray, calib_scores: np.ndarray, data: SplitData, protocol: Protocol,
    *, oracle_rates: tuple[float, ...] = (),
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Одна клетка: ``(метрики и единицы тревог, пособытийная таблица, строки ORACLE)``.

    Порог пересчитывается из score калибровки тем же путём, что в раннере
    (:func:`full_report` -> ``threshold_from_calibration``), и сверяется с ``runs.csv``.
    """
    rep = full_report(scores, data, calib_scores=calib_scores, **protocol.report_kwargs())
    thr = float(rep["threshold"])
    test = alarm_units_test(scores, data, thr, persistence=protocol.persistence,
                            node_reduce=protocol.node_reduce)
    cal = alarm_units_calib(calib_scores, data, thr, persistence=protocol.persistence,
                            node_reduce=protocol.node_reduce)
    row = {
        "threshold": thr, "padf": rep["padf"], "event_recall": rep["event_recall"],
        "alarms_per_hour": rep["alarms_per_hour"], "average_precision": rep["average_precision"],
        "roc_auc_reference_only": rep["roc_auc_reference_only"],
        "test_hours": test["hours"], "alarm_cells_per_hour": test["alarm_cells_per_hour"],
        "alarm_windows_per_hour": test["alarm_windows_per_hour"],
        "episodes_per_hour": test["episodes_per_hour"],
        "calib_hours": cal["hours"], "calib_budget_cells": protocol.alarm_budget_per_hour * cal["hours"],
        "calib_alarm_cells": cal["alarm_cells"], "calib_alarm_windows": cal["alarm_windows"],
        "calib_episodes": cal["episodes"],
    }
    events = event_table(scores, data, thr, half_life_min=protocol.half_life_min,
                         persistence=protocol.persistence, node_reduce=protocol.node_reduce)
    oracle = [oracle_report(scores, data, r, protocol) for r in oracle_rates]
    return row, events, oracle


def summarize_alarm_units(units: pd.DataFrame, nominal: float) -> pd.DataFrame:
    """Сводка по конфигурации: окна и эпизоды в час на тесте, опора порога в калибровке."""
    if units.empty:
        return pd.DataFrame(columns=["config", "n_cells", "nominal_per_hour"])
    g = units.groupby("config")
    out = g.agg(
        n_cells=("seed", "size"),
        windows_per_hour_mean=("alarm_windows_per_hour", "mean"),
        windows_per_hour_min=("alarm_windows_per_hour", "min"),
        windows_per_hour_max=("alarm_windows_per_hour", "max"),
        episodes_per_hour_mean=("episodes_per_hour", "mean"),
        episodes_per_hour_min=("episodes_per_hour", "min"),
        episodes_per_hour_max=("episodes_per_hour", "max"),
        calib_hours_mean=("calib_hours", "mean"),
        calib_windows_mean=("calib_alarm_windows", "mean"),
        calib_episodes_mean=("calib_episodes", "mean"),
        calib_episodes_min=("calib_episodes", "min"),
        calib_episodes_max=("calib_episodes", "max"),
    ).reset_index()
    out.insert(2, "nominal_per_hour", nominal)
    return out


# ================================================= B. случайный контроль
def random_null(
    datasets: dict[str, SplitData], seeds: range | list[int], protocol: Protocol,
    *, oracle_rates: tuple[float, ...] = (),
) -> pd.DataFrame:
    """Случайный score по многим сидам ровно так, как его использует сетка.

    ``RandomScorer(seed).score`` / ``.score_calib`` -> :func:`full_report`; сиды ``0..k-1``
    совпадают с клетками ``ctrl_random`` прогона бит в бит (проверяется в отчёте).
    """
    rows = []
    for ds_name, data in datasets.items():
        for seed in seeds:
            scorer = RandomScorer(seed=seed)
            s = np.asarray(scorer.score(data), dtype=np.float32)
            c = np.asarray(scorer.score_calib(data), dtype=np.float32)
            rep = full_report(s, data, calib_scores=c, **protocol.report_kwargs())
            units = alarm_units_test(s, data, rep["threshold"], persistence=protocol.persistence,
                                     node_reduce=protocol.node_reduce)
            row = {"dataset": ds_name, "seed": int(seed), "threshold": rep["threshold"],
                   "padf": rep["padf"], "event_recall": rep["event_recall"],
                   "alarms_per_hour": rep["alarms_per_hour"],
                   "alarm_windows_per_hour": units["alarm_windows_per_hour"],
                   "episodes_per_hour": units["episodes_per_hour"]}
            for r in oracle_rates:
                o = oracle_report(s, data, r, protocol)
                row[f"padf_ORACLE_{r:g}ph"] = o["padf_ORACLE"]
                row[f"alarms_per_hour_ORACLE_{r:g}ph"] = o["alarms_per_hour_ORACLE"]
            rows.append(row)
    return pd.DataFrame(rows)


def random_band(null: pd.DataFrame, n_draws: int, *, n_mc: int = 20_000, seed: int = 0,
                alpha: float = 0.05) -> dict[str, float]:
    """Полоса среднего прогона для случайного score (Monte Carlo).

    Каждый фолд вносит ``n_draws`` независимых значений из своего распределения
    (выборка с возвращением из ``null``), статистика — среднее по всем. ``n_draws``
    равно числу сидов прогона для конфигурации с независимыми сидами и 1 — для
    детерминированной (её сиды внутри фолда дают одно и то же число).
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(n_mc)
    folds = [g["padf"].to_numpy(dtype=float) for _, g in null.groupby("dataset", sort=True)]
    for vals in folds:
        total += vals[rng.integers(0, len(vals), size=(n_mc, n_draws))].sum(axis=1)
    means = total / (len(folds) * n_draws)
    return {"n_draws_per_fold": int(n_draws), "n_folds": len(folds), "n_mc": int(n_mc),
            "expected": float(null["padf"].mean()),
            "q_lo": float(np.quantile(means, alpha / 2)), "q_hi": float(np.quantile(means, 1 - alpha / 2))}


def per_block_quantiles(null: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    """Распределение padf одного блока: по каждому фолду и по всем фолдам вместе."""
    rows = []
    for name, vals in [*((d, g["padf"]) for d, g in null.groupby("dataset", sort=True)),
                       ("все фолды", null["padf"])]:
        v = vals.to_numpy(dtype=float)
        rows.append({"variant": f"один блок: {name}", "n_draws_per_fold": np.nan, "n_mc": np.nan,
                     "expected": float(v.mean()), "q_lo": float(np.quantile(v, alpha / 2)),
                     "q_hi": float(np.quantile(v, 1 - alpha / 2)), "n_values": len(v)})
    return pd.DataFrame(rows)


def classify_band(value: float, lo: float, hi: float) -> str:
    """Положение среднего относительно полосы случайного контроля."""
    if not np.isfinite(value):
        return "—"
    return "ниже" if value < lo else "выше" if value > hi else "внутри"


# ================================================== D. безмодельные score
def model_free_scores(X: np.ndarray, feature_names: list[str], kind: str) -> np.ndarray:
    """Безмодельный score ``[n, N]``: значение окна, размноженное по узлам.

    Последний шаг окна, среднее по сети стандартизованного признака (``X`` уже
    стандартизован загрузчиком по статистикам обучения). Размножение по узлам
    нужно, чтобы свёртка ``evaluation_view`` (max) вернула то же значение.
    """
    if kind not in MODEL_FREE_SCORES:
        raise ValueError(f"неизвестный безмодельный score {kind!r}; доступны: {sorted(MODEL_FREE_SCORES)}")
    feat, sign, _ = MODEL_FREE_SCORES[kind]
    if feat not in feature_names:
        raise ValueError(f"безмодельный score {kind!r} требует признак {feat!r}, "
                         f"а в данных только {list(feature_names)}")
    v = sign * X[:, :, -1, feature_names.index(feat)].mean(axis=1)
    return np.repeat(v[:, None], X.shape[1], axis=1).astype(np.float32)


def evaluate_model_free(datasets: dict[str, SplitData], protocol: Protocol) -> pd.DataFrame:
    """Метрики безмодельных score по тому же протоколу: порог — по калибровке ``X_calib``."""
    rows = []
    for ds_name, data in datasets.items():
        for kind in MODEL_FREE_SCORES:
            s = model_free_scores(data.X_test, data.feature_names, kind)
            c = model_free_scores(data.X_calib, data.feature_names, kind)
            rep = full_report(s, data, calib_scores=c, **protocol.report_kwargs())
            units = alarm_units_test(s, data, rep["threshold"], persistence=protocol.persistence,
                                     node_reduce=protocol.node_reduce)
            rows.append({"score": kind, "dataset": ds_name,
                         **{k: rep[k] for k in ("average_precision", "roc_auc_reference_only", "padf",
                                                "event_recall", "alarms_per_hour", "threshold")},
                         "episodes_per_hour": units["episodes_per_hour"]})
    return pd.DataFrame(rows)


# ============================================ E. псевдорепликация сидов
def seed_variance(runs: pd.DataFrame, *, metric: str = "padf", tol: float = 1e-12) -> pd.DataFrame:
    """Разброс ``metric`` по сидам внутри фолда и число различных значений.

    Нулевой разброс значит, что сиды одного фолда — одно наблюдение, повторённое
    ``n_seeds`` раз: блоков «фолд × сид» больше, чем независимых значений. При одном
    сиде в каждом фолде разброс не определён: флаг тогда ``NaN``, а не «нулевой».
    """
    per = (runs.groupby(["config", "dataset"])[metric]
           .agg(sd="std", n_distinct=lambda v: int(np.unique(np.round(v.to_numpy(float), 12)).size),
                n="size").reset_index())
    out = per.groupby("config").agg(
        n_folds=("dataset", "nunique"), n_seeds=("n", "max"),
        max_within_fold_sd=("sd", "max"), min_distinct=("n_distinct", "min"),
        max_distinct=("n_distinct", "max"), independent_values=("n_distinct", "sum"),
    ).reset_index()
    sd = out["max_within_fold_sd"]
    out["zero_within_fold_variance"] = (sd <= tol).astype(object).where(sd.notna(), np.nan)
    return out


def _zero_variance(seedvar: pd.DataFrame) -> list[str]:
    """Конфигурации с нулевым разбросом внутри фолда (неопределённый разброс сюда не входит)."""
    return seedvar.loc[seedvar["zero_within_fold_variance"].eq(True), "config"].tolist()


# ================================================ F. парные сравнения
def stratified_resample(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Индексы bootstrap ``[n_boot, n]``: внутри каждого слоя — выборка с возвращением своего размера.

    Число событий каждого фолда в каждой реплике сохраняется: фолды несопоставимы по
    числу событий, и без стратификации реплика могла бы состоять из одного фолда.
    """
    strata = np.asarray(strata)
    parts = []
    for s in pd.unique(strata):
        members = np.flatnonzero(strata == s)
        parts.append(members[rng.integers(0, len(members), size=(n_boot, len(members)))])
    return np.concatenate(parts, axis=1) if parts else np.zeros((n_boot, 0), dtype=np.int64)


def cluster_bootstrap(diff: np.ndarray, strata: np.ndarray, *, n_boot: int = 10_000,
                      seed: int = 0) -> dict[str, float]:
    """Кластерный bootstrap по событиям: единица — событие, слои — фолды.

    ``diff`` — разность кредитов (кредит уже усреднён по сидам внутри события).
    Статистика — средняя разность по всем событиям; p двусторонний.
    """
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return {"mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan,
                "half_width": np.nan, "n_events": 0}
    boots = d[stratified_resample(strata, n_boot, np.random.default_rng(seed))].mean(axis=1)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    p = min(1.0, 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean())))
    return {"mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p_value": p,
            "half_width": (hi - lo) / 2.0, "n_events": int(d.size)}


def event_credits(events: pd.DataFrame) -> pd.DataFrame:
    """Кредит события, усреднённый по сидам: ``config, dataset, event_id, credit, n_seeds``."""
    if events.empty:
        return pd.DataFrame(columns=["config", "dataset", "event_id", "credit", "n_seeds"])
    return (events.groupby(["config", "dataset", "event_id"])["credit"]
            .agg(credit="mean", n_seeds="size").reset_index())


def event_level_comparison(credits: pd.DataFrame, *, reference: str = REFERENCE, n_boot: int = 10_000,
                           seed: int = 0, n_seeds: int | None = None) -> pd.DataFrame:
    """Разность кредитов «конфигурация − референс» по событиям, кластерный bootstrap по фолдам.

    Сравнение полное (``event_complete``), если события те же, что у референса, и у
    обеих сторон кредит усреднён по одному и тому же числу сидов — ``n_seeds`` прогона,
    если он задан: среднее по одному сиду против среднего по пяти несопоставимо.
    """
    ref = credits[credits["config"] == reference][["dataset", "event_id", "credit", "n_seeds"]]
    rows = []
    if ref.empty:
        return pd.DataFrame(columns=["config", "event_mean_diff"])
    for cfg, g in credits[credits["config"] != reference].groupby("config"):
        m = g.merge(ref, on=["dataset", "event_id"], suffixes=("", "_ref"))
        res = cluster_bootstrap((m["credit"] - m["credit_ref"]).to_numpy(), m["dataset"].to_numpy(),
                                n_boot=n_boot, seed=seed)
        seeds_ok = bool((m["n_seeds"] == m["n_seeds_ref"]).all()) and (
            n_seeds is None or bool((m["n_seeds"] == n_seeds).all()))
        rows.append({"config": cfg, **{f"event_{k}": v for k, v in res.items()},
                     "event_n_folds": int(m["dataset"].nunique()),
                     "event_n_seeds_min": int(m["n_seeds"].min()) if len(m) else 0,
                     "event_n_seeds_ref_min": int(m["n_seeds_ref"].min()) if len(m) else 0,
                     "event_complete": len(m) == len(ref) and seeds_ok})
    return pd.DataFrame(rows)


def fold_level_comparison(runs: pd.DataFrame, *, reference: str = REFERENCE, metric: str = "padf",
                          n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Парный bootstrap по фолдам: сиды усреднены внутри фолда (одно значение на фолд).

    При четырёх фолдах у bootstrap всего 4^4 = 256 различных реплик: p = 0 значит лишь,
    что все фолды одного знака. Поэтому рядом — число фолдов с положительной разностью.
    """
    pivot = runs.pivot_table(index="dataset", columns="config", values=metric, aggfunc="mean")
    if reference not in pivot.columns:
        return pd.DataFrame(columns=["config", "fold_mean_diff"])
    base = pivot[reference].to_numpy(dtype=float)
    rows = []
    for cfg in pivot.columns:
        if cfg == reference:
            continue
        a = pivot[cfg].to_numpy(dtype=float)
        res = paired_bootstrap_test(a, base, n_boot=n_boot, seed=seed)
        d = (a - base)[np.isfinite(a) & np.isfinite(base)]
        rows.append({"config": cfg, **{f"fold_{k}": v for k, v in res.items()},
                     "fold_n_positive": int((d > 0).sum())})
    return pd.DataFrame(rows)


def pairwise_table(runs: pd.DataFrame, credits: pd.DataFrame, *, reference: str = REFERENCE,
                   n_boot: int = 10_000, n_seeds: int | None = None) -> pd.DataFrame:
    """Три схемы парного сравнения с референсом рядом: блоки, фолды, события.

    Вердикт ``compare_to_reference`` («лучше значимо») сюда не переносится: это
    диагностика протокола, а не вывод об архитектуре.
    """
    block = compare_to_reference(runs, metric="padf", reference=reference)
    if block.empty:
        return pd.DataFrame(columns=["config", "label"])
    block = block.drop(columns=["verdict"], errors="ignore")
    block = block.rename(columns={c: f"block_{c}" for c in block.columns if c not in ("config", "label")})
    fold = fold_level_comparison(runs, reference=reference, n_boot=n_boot)
    out = block.merge(fold, on="config", how="left")
    ev = event_level_comparison(credits, reference=reference, n_boot=n_boot, n_seeds=n_seeds)
    out = out.merge(ev, on="config", how="left") if not ev.empty else out.assign(event_mean_diff=np.nan)
    return out.reset_index(drop=True)


# ======================================================= вход-выход
def _load_saved(scores_dir: Path, config: str, dataset: str, seed: int, data: SplitData):
    """``(score теста, score калибровки)`` из ``scores/``; ``None``, если нет или формы чужие."""
    base = scores_dir / f"{config}__{dataset}__seed{seed}"
    f_test, f_calib = Path(f"{base}.npy"), Path(f"{base}__calib.npy")
    if not (f_test.exists() and f_calib.exists()):
        return None
    s, c = np.load(f_test), np.load(f_calib)
    if data.X_calib is None or s.shape != data.y_test.shape or c.shape != (len(data.X_calib), data.n_nodes):
        return None
    return s, c


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "да" if v else "нет"
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _independent(seedvar: pd.DataFrame, config: str):
    """Колонка «сиды независимы»: да / нет / не определено (один сид в фолде)."""
    if config not in seedvar.index:
        return "не определено"
    v = seedvar.loc[config, "zero_within_fold_variance"]
    return "не определено" if not isinstance(v, (bool, np.bool_)) else not bool(v)


def _is_true(v) -> bool:
    """Флаг строки таблицы: ``NaN`` (нет данных) — не истина."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def _ci(d: float, lo: float, hi: float) -> str:
    return "—" if not np.isfinite(d) else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _seeds_text(seeds) -> str:
    s = sorted(int(v) for v in seeds)
    return f"{s[0]}–{s[-1]}" if len(s) > 1 and s == list(range(s[0], s[-1] + 1)) else ", ".join(map(str, s))


# ====================================================== сверка с ТЗ
def _tz_text(ref: float, tol: float) -> str:
    """Число ТЗ с той точностью, с которой оно записано (допуск — половина последнего знака)."""
    if tol <= 0:
        return f"{ref:g}"
    digits = max(0, int(round(-np.log10(2.0 * tol))))
    return f"{ref:.{digits}f}"


def tz_reconciliation(computed: dict[str, float | None], band_rows: dict[str, tuple[float, float]],
                      zero_var: list[str] | None, *,
                      computed_notes: dict[str, str] | None = None) -> pd.DataFrame:
    """Таблица «ТЗ v2 §1 -> пересчёт». ``None`` в ``computed`` — числа нет (нет score).

    ``zero_var=None`` — разброс по сидам не определён (один сид в каждом фолде).
    ``computed_notes`` — пояснение к пересчитанному числу (например, сколько сидов).
    """
    notes = computed_notes or {}
    rows = []
    for key, (desc, ref, tol) in TZ_REFERENCE.items():
        got = computed.get(key)
        if got is None or not np.isfinite(got):
            status, diff, text = "нет score", np.nan, "—"
        else:
            diff = abs(float(got) - float(ref))
            text = f"{got:.4f}" + (f" ({notes[key]})" if key in notes else "")
            ok = diff <= tol + 1e-9
            status = ("расходится" if not ok else "совпадает до 3-го знака" if tol <= 0.0005
                      else f"совпадает с точностью ТЗ (±{tol:g})")
        rows.append({"пункт": desc, "ТЗ": _tz_text(ref, tol), "пересчёт": text, "abs diff": diff,
                     "статус": status})
    lo_tz, hi_tz = TZ_RANDOM_BAND
    for i, (name, ref) in enumerate((("нижняя", lo_tz), ("верхняя", hi_tz))):
        vals = {k: v[i] for k, v in band_rows.items()}
        text = "; ".join(f"{k}: {v:.4f}" for k, v in vals.items())
        between = len(vals) == 2 and min(vals.values()) <= ref <= max(vals.values())
        rows.append({"пункт": f"полоса случайного контроля, {name} граница", "ТЗ": f"{ref:.3f}",
                     "пересчёт": text, "abs diff": np.nan,
                     "статус": "между вариантами (определение в ТЗ не указано)" if between else "расходится"})
    if zero_var is None:
        rows.append({"пункт": "нулевой разброс по сидам внутри фолда", "ТЗ": ", ".join(TZ_ZERO_VARIANCE),
                     "пересчёт": "—", "abs diff": np.nan,
                     "статус": "не определено (один сид в каждом фолде)"})
    else:
        same = set(zero_var) == set(TZ_ZERO_VARIANCE)
        rows.append({"пункт": "нулевой разброс по сидам внутри фолда", "ТЗ": ", ".join(TZ_ZERO_VARIANCE),
                     "пересчёт": ", ".join(sorted(zero_var)) or "нет", "abs diff": np.nan,
                     "статус": "совпадает" if same else "расходится"})
    return pd.DataFrame(rows)


# ================================================ сквозная диагностика
@dataclass
class _Cells:
    units: pd.DataFrame
    events: pd.DataFrame
    oracle: pd.DataFrame
    reproduction: pd.DataFrame
    missing: pd.DataFrame


def _diagnose_cells(
    runs: pd.DataFrame, datasets: dict[str, SplitData], run_dir: Path, out: Path, protocol: Protocol,
    *, grid: dict, cfg: ExperimentConfig, param_budget: int, oracle_rates: tuple[float, ...],
    recompute_nontrainable: bool, device: str, log: Callable[[str], None],
) -> _Cells:
    """Клетки ``runs.csv``: сохранённые score, пересчёт небучаемых или пропуск с причиной."""
    scores_dir = run_dir / "scores"
    rec_dir = out / "scores_recomputed"
    train_cfg = cfg.train_config()
    train_cfg.device, train_cfg.progress = device, False
    units, events, oracle, repro, missing = [], [], [], [], []
    cells = runs.sort_values(["dataset", "seed", "config"])
    for r in cells.itertuples():
        data = datasets[r.dataset]
        key = {"config": r.config, "dataset": r.dataset, "seed": int(r.seed)}
        got = _load_saved(scores_dir, r.config, r.dataset, int(r.seed), data)
        source = "saved"
        if got is None:
            c = grid.get(r.config)
            if c is None:
                missing.append({**key, "reason": "конфигурации нет в сетке конфига"})
                continue
            if not recompute_nontrainable or (c.is_trainable and not c.randomize_only):
                missing.append({**key, "reason": "обучаемая: нужен scores/ прогона" if c.is_trainable
                                and not c.randomize_only else "нет scores/ (пересчёт не запрошен)"})
                continue
            log(f"пересчёт {r.config} {r.dataset} seed={r.seed}")
            # раннер строит сеть до set_seed внутри обучения: без этого веса необученной сети
            # зависели бы от того, какие клетки пересчитаны раньше в этом же вызове
            set_seed(int(r.seed))
            _, _, s, _, cs = run_config(
                c, data, seed=int(r.seed), dataset_name=r.dataset, train_cfg=train_cfg,
                param_budget=param_budget, alarm_budget_per_hour=protocol.alarm_budget_per_hour,
                half_life_min=protocol.half_life_min, persistence=protocol.persistence,
                node_reduce=protocol.node_reduce, checkpoint_dir=None,
            )
            rec_dir.mkdir(parents=True, exist_ok=True)
            np.save(rec_dir / f"{r.config}__{r.dataset}__seed{r.seed}.npy", s)
            np.save(rec_dir / f"{r.config}__{r.dataset}__seed{r.seed}__calib.npy", cs)
            got, source = (s, cs), "recomputed"
        row, ev, orc = cell_diagnostics(got[0], got[1], data, protocol, oracle_rates=oracle_rates)
        units.append({**key, "source": source, "nominal_per_hour": protocol.alarm_budget_per_hour,
                      "threshold_runs": float(r.threshold), "alarms_per_hour_runs": float(r.alarms_per_hour),
                      **row})
        events.append(ev.assign(**key))
        oracle.extend({**key, **o} for o in orc)
        thr_tol = 1e-4 * max(1.0, abs(float(r.threshold)))
        padf_ok = abs(float(r.padf) - row["padf"]) <= 1e-9
        thr_ok = abs(float(r.threshold) - row["threshold"]) <= thr_tol
        aph_ok = abs(float(r.alarms_per_hour) - row["alarm_cells_per_hour"]) <= 1e-9
        repro.append({
            **key, "source": source,
            "expected_bit_identical": not (source == "recomputed" and grid[r.config].randomize_only),
            "padf_runs": float(r.padf), "padf_recomputed": row["padf"],
            "abs_diff_padf": abs(float(r.padf) - row["padf"]),
            "threshold_runs": float(r.threshold), "threshold_recomputed": row["threshold"],
            "abs_diff_threshold": abs(float(r.threshold) - row["threshold"]),
            "alarms_per_hour_runs": float(r.alarms_per_hour),
            "alarm_cells_per_hour": row["alarm_cells_per_hour"],
            "abs_diff_alarms_per_hour": abs(float(r.alarms_per_hour) - row["alarm_cells_per_hour"]),
            "average_precision_runs": float(r.average_precision),
            "average_precision_recomputed": row["average_precision"],
            "padf_match": padf_ok, "threshold_match": thr_ok,
            # score клетки — те же, что в прогоне: совпали и padf, и порог, и частота тревог
            "reproduced": padf_ok and thr_ok and aph_ok,
        })
    unit_cols = ["config", "dataset", "seed", "source"]
    return _Cells(
        units=pd.DataFrame(units, columns=None if units else unit_cols),
        events=pd.concat(events, ignore_index=True) if events else pd.DataFrame(
            columns=["config", "dataset", "seed", "event_id", "credit"]),
        oracle=pd.DataFrame(oracle, columns=None if oracle else [*unit_cols[:3], "rate_per_hour"]),
        reproduction=pd.DataFrame(repro, columns=None if repro else unit_cols),
        missing=pd.DataFrame(missing, columns=None if missing else ["config", "dataset", "seed", "reason"]),
    )


def run_diagnostics(
    cfg: ExperimentConfig,
    datasets: dict[str, SplitData],
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    config_path: str | None = None,
    random_seeds: int = 200,
    n_boot: int = 10_000,
    n_mc: int = 20_000,
    oracle_rates: tuple[float, ...] = (0.25, 1.0),
    recompute_nontrainable: bool = False,
    tz_reference: bool = False,
    device: str = "cpu",
    log: Callable[[str], None] | None = None,
) -> Path:
    """Посчитать диагностику прогона и записать ``DIAGNOSTICS.md`` и CSV в ``out_dir``.

    ``datasets`` — уже собранные датасеты прогона (``build_datasets(cfg)``; можно
    подмножество фолдов — тогда ``runs.csv`` фильтруется по нему). Нужен только
    ``runs.csv``; ``manifest.json``, ``events.csv`` и ``scores/`` — по наличию.
    Небучаемые клетки без score пересчитываются при ``recompute_nontrainable``.
    """
    log = log or (lambda _msg: None)
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_all = pd.read_csv(run_dir / "runs.csv")
    runs = runs_all[runs_all["dataset"].isin(list(datasets))].reset_index(drop=True)
    if runs.empty:
        raise ValueError(f"в {run_dir / 'runs.csv'} нет строк для датасетов {sorted(datasets)}")
    unknown = sorted(set(runs_all["dataset"]) - set(datasets))
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    bad = dataset_mismatches(datasets, manifest)
    if bad:
        raise ValueError("датасеты конфига не совпадают с датасетами прогона (manifest.json): "
                         + "; ".join(bad) + ". Конфиг и --run-dir от разных прогонов?")
    protocol, notes = protocol_for_run(cfg, manifest)
    param_budget = int(manifest.get("param_budget", cfg.param_budget))
    run_seeds = sorted(int(s) for s in runs["seed"].unique())
    grid = {c.name: c for c in get_grid(cfg.grid)}

    log(f"клетки: {len(runs)}")
    cells = _diagnose_cells(runs, datasets, run_dir, out, protocol, grid=grid, cfg=cfg,
                            param_budget=param_budget, oracle_rates=tuple(oracle_rates),
                            recompute_nontrainable=recompute_nontrainable, device=device, log=log)
    ev_path = run_dir / "events.csv"
    if ev_path.exists():
        ev_all = pd.read_csv(ev_path)
        events, credit_source = ev_all[ev_all["dataset"].isin(list(datasets))], "events.csv прогона"
    else:
        events, credit_source = cells.events, "пересчёт по score (eventlog.event_table)"
    credits = event_credits(events)

    log(f"случайный контроль: {random_seeds} сидов × {len(datasets)} фолдов")
    null = random_null(datasets, range(random_seeds), protocol, oracle_rates=tuple(oracle_rates))
    k = len(run_seeds)
    band_k = random_band(null, k, n_mc=n_mc)
    band_1 = random_band(null, 1, n_mc=n_mc)
    band = pd.concat([pd.DataFrame([{"variant": f"{k} на фолд (сиды независимы)", **band_k},
                                    {"variant": "1 на фолд (детерминированная конфигурация)", **band_1}]),
                      per_block_quantiles(null)], ignore_index=True)
    rnd_runs = runs[runs["config"] == "ctrl_random"][["dataset", "seed", "padf", "threshold"]]
    null = null.merge(rnd_runs.rename(columns={"padf": "padf_runs", "threshold": "threshold_runs"}),
                      on=["dataset", "seed"], how="left")
    null["in_run"] = null["padf_runs"].notna()

    log("безмодельные score")
    model_free = evaluate_model_free(datasets, protocol)
    seedvar = seed_variance(runs)
    log("парные сравнения")
    pairwise = pairwise_table(runs, credits, n_boot=n_boot, n_seeds=len(run_seeds))
    summary = summarize_alarm_units(cells.units, protocol.alarm_budget_per_hour)

    for name, df in (("alarm_units", cells.units), ("alarm_units_summary", summary), ("random_null", null),
                     ("random_band", band), ("model_free", model_free), ("seed_variance", seedvar),
                     ("pairwise", pairwise), ("reproduction", cells.reproduction)):
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    oracle = pd.concat(
        [null[["dataset", "seed"]].assign(config="ctrl_random[null]", rate_per_hour=float(r),
                                          padf_ORACLE=null[f"padf_ORACLE_{r:g}ph"],
                                          alarms_per_hour_ORACLE=null[f"alarms_per_hour_ORACLE_{r:g}ph"])
         for r in oracle_rates] + [cells.oracle],
        ignore_index=True,
    )
    oracle.to_csv(out / "oracle_rates.csv", index=False, encoding="utf-8")

    ctx = {
        "cfg": cfg, "config_path": config_path, "run_dir": run_dir, "out": out, "manifest": manifest,
        "protocol": protocol, "notes": notes, "runs": runs, "run_seeds": run_seeds, "datasets": datasets,
        "unknown_datasets": unknown, "cells": cells, "summary": summary, "null": null, "band": band,
        "band_k": band_k, "band_1": band_1, "model_free": model_free, "seedvar": seedvar,
        "pairwise": pairwise, "credits": credits, "credit_source": credit_source, "oracle": oracle,
        "oracle_rates": tuple(oracle_rates), "random_seeds": random_seeds, "n_boot": n_boot, "n_mc": n_mc,
        "grid": grid, "tz_reference": tz_reference, "credits_from_events_csv": ev_path.exists(),
    }
    path = out / "DIAGNOSTICS.md"
    path.write_text(_render(ctx), encoding="utf-8")
    return path


# ============================================================ отчёт
def block_ci_mc_range(runs: pd.DataFrame, config: str, *, reference: str = REFERENCE,
                      seeds: range = range(10),
                      n_boots: tuple[int, ...] = (2000, 5000, 10_000, 20_000)) -> dict:
    """Разброс границ блочного ДИ по сидам и числу повторов bootstrap (как ``compare_to_reference``).

    Границы bootstrap — оценка Monte Carlo; диапазон показывает, насколько они
    сдвигаются от сида и ``n_boot``, чтобы расхождение с ТЗ можно было отнести к шуму или нет.
    """
    pivot = runs.pivot_table(index="block", columns="config", values="padf", aggfunc="mean")
    if config not in pivot.columns or reference not in pivot.columns:
        return {}
    a, b = pivot[config].to_numpy(dtype=float), pivot[reference].to_numpy(dtype=float)
    res = [paired_bootstrap_test(a, b, n_boot=nb, seed=s) for nb in n_boots for s in seeds]
    lo, hi = [r["ci_lo"] for r in res], [r["ci_hi"] for r in res]
    return {"seeds": seeds, "n_boots": tuple(n_boots), "lo_min": min(lo), "lo_max": max(lo),
            "hi_min": min(hi), "hi_max": max(hi)}


def _mc_bound_text(name: str, ref: float, tol: float, lo: float, hi: float) -> str:
    """Попадает ли число ТЗ (с точностью записи ``±tol``) в наблюдаемый разброс границы."""
    if lo - tol <= ref <= hi + tol:
        return (f"{name} граница ТЗ {_tz_text(ref, tol)} попадает в разброс [{lo:.4f}, {hi:.4f}] — "
                "расхождение объясняется шумом Monte Carlo")
    return (f"{name} граница ТЗ {_tz_text(ref, tol)} не воспроизводится ни при одном опробованном сиде и "
            f"числе повторов (наблюдаемый разброс [{lo:.4f}, {hi:.4f}]): в ТЗ, вероятно, другое вычисление "
            "(какое — неизвестно)")


def _tz_computed(ctx: dict) -> tuple[dict[str, float | None], dict[str, tuple[float, float]], list[str]]:
    """Величины для сверки с ТЗ. Требующие score всех конфигураций — ``None``, если покрытие неполное."""
    runs, cells, null, ds = ctx["runs"], ctx["cells"], ctx["null"], ctx["datasets"]
    means = runs.groupby("config")[["padf", "average_precision", "roc_auc_reference_only",
                                    "alarms_per_hour"]].mean()
    corridor = all(d.meta.get("labels_are_corridor_level") for d in ds.values())
    cal_hours = [len(d.t_calib) * float(d.meta.get("step_min", 0.5)) / 60.0 for d in ds.values()]
    bud = [np.floor(ctx["protocol"].alarm_budget_per_hour * h + 1e-9) for h in cal_hours]
    complete = cells.missing.empty
    c: dict[str, float | None] = {
        "n_events_test": float(sum(len(d.events) for d in ds.values())),
        "random_expected_padf": float(null["padf"].mean()),
        "calib_hours_mean": float(np.mean(cal_hours)),
        "calib_budget_windows_min": float(min(bud)), "calib_budget_windows_max": float(max(bud)),
    }
    for key in TZ_REFERENCE:
        for prefix, col in (("padf_mean.", "padf"), ("ap_mean.", "average_precision"),
                            ("roc_mean.", "roc_auc_reference_only")):
            if key.startswith(prefix):
                name = key[len(prefix):]
                c[key] = float(means.loc[name, col]) if name in means.index else None
    # после свёртки по узлам (корридор-уровневые метки) ячейка тревоги = окно
    if corridor:
        per_fold = runs.groupby(["config", "dataset"])["alarms_per_hour"].mean()
        c["alarm_windows_per_hour_fold_min"] = float(per_fold.min())
        c["alarm_windows_per_hour_fold_max"] = float(per_fold.max())
        c["alarm_windows_per_hour_config_min"] = float(means["alarms_per_hour"].min())
        c["alarm_windows_per_hour_config_max"] = float(means["alarms_per_hour"].max())
    if complete and not cells.units.empty:
        ep = cells.units.groupby("config")["episodes_per_hour"].mean()
        c["episodes_per_hour_config_min"] = float(ep.min())
        c["episodes_per_hour_config_max"] = float(ep.max())
        c["calib_episodes_min"] = float(cells.units["calib_episodes"].min())
        c["calib_episodes_max"] = float(cells.units["calib_episodes"].max())
    orc = cells.oracle
    n_rnd = int((runs["config"] == "ctrl_random").sum())
    if not orc.empty:
        rnd = orc[(orc["config"] == "ctrl_random") & np.isclose(orc["rate_per_hour"].astype(float), 1.0)]
        if len(rnd) and len(rnd) == n_rnd:
            c["random_oracle_padf_1"] = float(rnd["padf_ORACLE"].mean())
    # без score клеток: те же сиды есть в нулевом распределении — если оно воспроизводит runs.csv
    if "random_oracle_padf_1" not in c and "padf_ORACLE_1ph" in null.columns:
        own = null[null["in_run"]]
        same = not own.empty and float((own["padf"] - own["padf_runs"]).abs().max()) <= 1e-9
        if n_rnd and len(own) == n_rnd and same:
            c["random_oracle_padf_1"] = float(own["padf_ORACLE_1ph"].mean())
    mf = ctx["model_free"].groupby("score")[["average_precision", "roc_auc_reference_only"]].mean()
    if "occ_mean" in mf.index:
        c["model_free.occ_mean.average_precision"] = float(mf.loc["occ_mean", "average_precision"])
        c["model_free.occ_mean.roc_auc"] = float(mf.loc["occ_mean", "roc_auc_reference_only"])
    pw = ctx["pairwise"].set_index("config") if not ctx["pairwise"].empty else pd.DataFrame()
    if "base_iforest" in pw.index:
        c["block_ci_lo.base_iforest"] = float(pw.loc["base_iforest", "block_ci_lo"])
        c["block_ci_hi.base_iforest"] = float(pw.loc["base_iforest", "block_ci_hi"])
        it = pw.loc["base_iforest"]
        if pd.notna(it.get("event_mean_diff", np.nan)) and _is_true(it.get("event_complete")):
            c["event_ci_lo.base_iforest"] = float(pw.loc["base_iforest", "event_ci_lo"])
            c["event_ci_hi.base_iforest"] = float(pw.loc["base_iforest", "event_ci_hi"])
    # доступность по событиям определяют кредиты (events.csv или score), а не покрытие score
    if "event_complete" in pw.columns and len(pw) and pw["event_complete"].eq(True).all():
        c["event_half_width_min"] = float(pw["event_half_width"].min())
        c["event_half_width_max"] = float(pw["event_half_width"].max())
    bk, b1 = ctx["band_k"], ctx["band_1"]
    band_rows = {f"{bk['n_draws_per_fold']} на фолд": (bk["q_lo"], bk["q_hi"]),
                 "1 на фолд": (b1["q_lo"], b1["q_hi"])}
    return c, band_rows, _zero_variance(ctx["seedvar"])


def _render(ctx: dict) -> str:
    cfg, runs, cells, protocol = ctx["cfg"], ctx["runs"], ctx["cells"], ctx["protocol"]
    manifest, null, pairwise = ctx["manifest"], ctx["null"], ctx["pairwise"]
    labels = runs.groupby("config")["label"].first()
    n_saved = int((cells.units["source"] == "saved").sum()) if not cells.units.empty else 0
    n_rec = int((cells.units["source"] == "recomputed").sum()) if not cells.units.empty else 0
    sha = manifest.get("environment", {}).get("git_sha", "неизвестен (нет manifest.json)")
    k = len(ctx["run_seeds"])
    L: list[str] = ["# Диагностика протокола прогона\n"]
    L.append(
        f"Прогон: `{ctx['run_dir']}`, коммит прогона `{sha}`. Конфиг: `{ctx['config_path'] or cfg.name}` "
        f"(сетка `{cfg.grid}`). Фолды: {', '.join(ctx['datasets'])}; сиды {_seeds_text(ctx['run_seeds'])} "
        f"({k}). Протокол: бюджет {protocol.alarm_budget_per_hour:g} подтверждённой ложной тревоги/ч, "
        f"подтверждение {protocol.persistence} окна, полупериод {protocol.half_life_min:g} мин, "
        f"свёртка по узлам `{protocol.node_reduce}`.\n")
    L.append(f"Клеток в `runs.csv`: **{len(runs)}**; с сохранёнными score: **{n_saved}**; пересчитано "
             f"(небучаемые): **{n_rec}**; без score: **{len(cells.missing)}**.\n")
    if ctx["unknown_datasets"]:
        L.append(f"Фолды прогона вне диагностики (`--datasets`): {', '.join(ctx['unknown_datasets'])}.\n")
    if ctx["notes"]:
        L.extend([f"- {n}" for n in ctx["notes"]] + [""])
    L.append("Это диагностика протокола: выводов об архитектурах здесь нет. Величины с пометкой "
             "**ORACLE** подбирают порог по тестовым меткам и в ранжировании не используются.\n")

    # ---------------------------------------------------------------- A
    L.append("## A. Единица бюджета тревог: ячейки, окна, эпизоды\n")
    L.append("Ячейка ложной тревоги — `confirm_alarms(s > thr) & (y == 0)` после свёртки по узлам, ровно "
             "числитель `alarms_per_hour`. Окно тревоги — окно хотя бы с одной такой ячейкой. Эпизод — "
             "начало непрерывной серии окон тревоги внутри сегмента (без склейки через разрыв). Порог "
             "пересчитан из score калибровки так же, как в раннере.\n")
    if cells.units.empty:
        L.append("Нет ни одной клетки со score — раздел не посчитан (см. «Чего не хватает»).\n")
    else:
        s = ctx["summary"].merge(
            cells.reproduction.groupby("config")["reproduced"].all().rename("score = прогона").reset_index(),
            on="config", how="left")
        L.append(_md(s[["config", "n_cells", "nominal_per_hour", "windows_per_hour_mean",
                        "windows_per_hour_min", "windows_per_hour_max", "episodes_per_hour_mean",
                        "episodes_per_hour_min", "episodes_per_hour_max", "score = прогона"]], 3))
        L.append("\n«score = прогона: нет» — у какой-то клетки не совпали padf, порог или частота тревог "
                 "с `runs.csv`: числа этой строки посчитаны по другим score (раздел «Воспроизведение»).\n")
        L.append("На чём держится порог (калибровочная выборка, тот же порог):\n")
        L.append(_md(s[["config", "calib_hours_mean", "calib_windows_mean", "calib_episodes_mean",
                        "calib_episodes_min", "calib_episodes_max", "score = прогона"]], 2))
        L.append(f"\nБюджет калибровки в окнах: {protocol.alarm_budget_per_hour:g}/ч × часы калибровки. "
                 "Пособытийно по клеткам — `alarm_units.csv`.\n")
    m = runs.groupby("config")["alarms_per_hour"].agg(["mean", "min", "max"]).reset_index()
    L.append("По `runs.csv` (без score; при корридор-уровневых метках ячейка = окно) — ячеек "
             "тревоги в час:\n")
    L.append(_md(m.rename(columns={"mean": "среднее", "min": "мин", "max": "макс"}), 2))
    L.append("")

    # ---------------------------------------------------------------- B
    bk, b1, band = ctx["band_k"], ctx["band_1"], ctx["band"]
    L.append("## B. Случайный контроль: нулевое распределение и полоса\n")
    chk = null[null["in_run"]]
    if chk.empty:
        L.append("Сидов `ctrl_random` прогона среди пересчитанных нет — проверка воспроизведения "
                 "не выполнена.\n")
    else:
        dp = float((chk["padf"] - chk["padf_runs"]).abs().max())
        dt = float((chk["threshold"] - chk["threshold_runs"]).abs().max())
        verdict = "совпадают" if dp <= 1e-9 and dt <= 1e-6 else "РАСХОДЯТСЯ"
        L.append(f"Проверка: сиды {_seeds_text(chk['seed'].unique())} нулевого распределения против "
                 f"строк `ctrl_random` в `runs.csv` ({len(chk)} клеток) — {verdict}: max |Δpadf| = "
                 f"{dp:.2e}, max |Δпорог| = {dt:.2e}.\n")
    L.append(f"{ctx['random_seeds']} сидов × {len(ctx['datasets'])} фолдов. Ожидаемый padf случайного "
             f"score (среднее по фолдам и сидам): **{bk['expected']:.5f}**; эпизодов/ч в среднем "
             f"{null['episodes_per_hour'].mean():.3f}, окон тревоги/ч "
             f"{null['alarm_windows_per_hour'].mean():.3f}.\n")
    L.append(f"Полоса 95% для среднего прогона (Monte Carlo, {ctx['n_mc']} повторов, фиксированный сид):\n")
    L.append(_md(band[["variant", "n_draws_per_fold", "expected", "q_lo", "q_hi"]].assign(
        n_draws_per_fold=band["n_draws_per_fold"].map(lambda v: "—" if pd.isna(v) else str(int(v)))), 4))
    lo_tz, hi_tz = TZ_RANDOM_BAND
    between = b1["q_lo"] <= lo_tz <= bk["q_lo"] and bk["q_hi"] <= hi_tz <= b1["q_hi"]
    L.append(f"\nВариант «{k} на фолд» верен для конфигурации, у которой сиды независимы; «1 на фолд» — "
             "для детерминированной (её сиды внутри фолда дают одно число, раздел E). ТЗ v2 §1.2 даёт "
             f"[{lo_tz:.3f}, {hi_tz:.3f}] без определения полосы; этот интервал "
             + ("лежит между двумя вариантами" if between else "НЕ лежит между двумя вариантами")
             + ". Классификация конфигураций у границы зависит от определения полосы.\n")
    sv = ctx["seedvar"].set_index("config")
    means = runs.groupby("config")["padf"].mean().sort_values(ascending=False)
    cls = pd.DataFrame({
        "config": means.index, "label": [labels[c] for c in means.index], "padf (runs.csv)": means.values,
        f"полоса {k} на фолд": [classify_band(v, bk["q_lo"], bk["q_hi"]) for v in means.values],
        "полоса 1 на фолд": [classify_band(v, b1["q_lo"], b1["q_hi"]) for v in means.values],
        "сиды независимы": [_independent(sv, c) for c in means.index],
    })
    L.append(_md(cls, 3))
    L.append("")

    # ---------------------------------------------------------------- C
    L.append("## C. ORACLE: равная реальная частота тревог (только диагностика)\n")
    L.append(f"{_ORACLE_NOTE} Порог даёт не более заданного числа подтверждённых ложных ячеек в час на "
             "тесте (при корридор-уровневых метках ячейка = окно), одинаково для всех score, и показывает, "
             "что метрика делает при равной реальной частоте.\n")
    orc = ctx["oracle"]
    if not orc.empty:
        t = (orc.groupby(["config", "rate_per_hour"])["padf_ORACLE"].agg(["mean", "size"]).reset_index()
             .pivot(index="config", columns="rate_per_hour", values="mean"))
        t.columns = [f"padf_ORACLE при {c:g}/ч" for c in t.columns]
        # порядок фиксированный (опорная строка, затем порядок сетки), не по значению: это не рейтинг
        order = ["ctrl_random[null]", *ctx["grid"]]
        t = t.reindex([c for c in order if c in t.index] + sorted(set(t.index) - set(order)))
        t.index.name = "config"
        ok = cells.reproduction.groupby("config")["reproduced"].all() if not cells.reproduction.empty \
            else pd.Series(dtype=bool)
        t["score = прогона"] = [True if c == "ctrl_random[null]" else bool(ok.get(c, False)) for c in t.index]
        L.append(_md(t.reset_index(), 3))
        L.append("\n`ctrl_random[null]` — среднее по всем сидам нулевого распределения. Остальные "
                 "строки — клетки со score (сохранёнными или пересчитанными). Порядок строк — порядок "
                 "сетки, а не значение.\n")

    # ---------------------------------------------------------------- D
    mf = ctx["model_free"]
    L.append("## D. Безмодельные score\n")
    L.append("Score окна — среднее по сети стандартизованного признака на последнем шаге окна, "
             "без модели: " +"; ".join(f"`{k}` — {v[2]}" for k, v in MODEL_FREE_SCORES.items())
             + ". Порог — по калибровке `X_calib`, как у моделей.\n")
    agg = mf.groupby("score")[["average_precision", "roc_auc_reference_only", "padf", "alarms_per_hour",
                                "episodes_per_hour"]].mean().reset_index()
    L.append(_md(agg, 3))
    cmp_rows = [{"источник": "без модели", "config": r.score, "AP": r.average_precision,
                 "ROC (справочно)": r.roc_auc_reference_only} for r in agg.itertuples()]
    rm = runs.groupby("config")[["average_precision", "roc_auc_reference_only"]].mean()
    cmp_rows += [{"источник": "runs.csv", "config": c, "AP": r.average_precision,
                  "ROC (справочно)": r.roc_auc_reference_only} for c, r in rm.iterrows()]
    L.append("\nСопоставление поточечной разделимости (средние по фолдам и сидам):\n")
    L.append(_md(pd.DataFrame(cmp_rows).sort_values("AP", ascending=False), 3))
    L.append("")

    # ---------------------------------------------------------------- E
    L.append("## E. Псевдорепликация сидов\n")
    L.append("Разброс padf по сидам внутри фолда (SD, максимум по фолдам) и число различных значений. "
             "Нулевой разброс: блоков «фолд × сид» больше, чем независимых наблюдений.\n")
    L.append(_md(ctx["seedvar"], 4))
    zero = _zero_variance(ctx["seedvar"])
    L.append(f"\nНулевой разброс внутри фолда: {', '.join(zero) if zero else 'нет'}.\n")
    undef = ctx["seedvar"].loc[ctx["seedvar"]["zero_within_fold_variance"].isna(), "config"].tolist()
    if undef:
        L.append(f"Разброс не определён (один сид в каждом фолде): {', '.join(undef)}.\n")

    # ---------------------------------------------------------------- F
    L.append(f"## F. Парные сравнения с референсом `{REFERENCE}`\n")
    L.append("Разность padf «конфигурация − референс», 95% ДИ, двусторонний p. Блоки — "
             f"`report.compare_to_reference` ({runs['block'].nunique()} блоков «фолд × сид», 5000 повторов, "
             f"как в RESULTS.md); фолды — сиды усреднены внутри фолда ({ctx['n_boot']} повторов); события — "
             "кластерный bootstrap, кредит события усреднён по сидам, ресэмплинг стратифицирован по фолду "
             f"({ctx['n_boot']} повторов). Кредиты: {ctx['credit_source']}.\n")
    if pairwise.empty:
        L.append("Референса нет в прогоне — раздел не посчитан.\n")
    else:
        rows = []
        for r in pairwise.to_dict("records"):
            def g(col: str, r=r) -> float:     # колонок событий может не быть
                return r.get(col, np.nan)
            rows.append({
                "config": r["config"],
                "блоки Δ [ДИ]": _ci(g("block_mean_diff"), g("block_ci_lo"), g("block_ci_hi")),
                "p блоки": g("block_p_value"),
                "фолды Δ [ДИ]": _ci(g("fold_mean_diff"), g("fold_ci_lo"), g("fold_ci_hi")),
                "p фолды": g("fold_p_value"),
                "фолдов Δ>0": (f"{int(g('fold_n_positive'))} из {int(g('fold_n'))}"
                               if np.isfinite(g("fold_n")) and g("fold_n") > 0 else "—"),
                "события Δ [ДИ]": _ci(g("event_mean_diff"), g("event_ci_lo"), g("event_ci_hi")),
                "p события": g("event_p_value"),
                "полуширина (события)": g("event_half_width"),
                "событий": g("event_n_events"),
            })
        L.append(_md(pd.DataFrame(rows), 3))
        n_folds = runs["dataset"].nunique()
        L.append(f"\nФолдовый bootstrap идёт по {n_folds} значениям: различных реплик всего "
                 f"{n_folds}^{n_folds} = {n_folds ** n_folds}, p = 0 значит лишь, что все фолды одного "
                 "знака, "
                 "а перцентильный ДИ занижает неопределённость. Точный двусторонний p перестановки знаков "
                 f"при {n_folds} фолдах не бывает меньше {2.0 / 2 ** n_folds:.3f}; поэтому рядом — число "
                 "фолдов с положительной разностью.\n")
        inc = (sorted(pairwise.loc[pairwise["event_complete"].eq(False), "config"])
               if "event_complete" in pairwise.columns else [])
        if inc:
            L.append("Сравнение по событиям неполное (не те события или другое число сидов, чем у "
                     f"референса и в прогоне; см. `pairwise.csv`, `event_n_seeds_*`): {', '.join(inc)}.\n")
        hw = pairwise.get("event_half_width", pd.Series(dtype=float))
        if "event_complete" in pairwise.columns:
            hw = hw[pairwise["event_complete"].eq(True)]           # неполные пары в MDE не входят
        hw = hw.dropna()
        if hw.empty:
            L.append("Минимально различимый эффект (полуширина ДИ по событиям) не посчитан: нет полных "
                     "кредитов референса или пар.\n")
        else:
            L.append(f"**Минимально различимый эффект** (полуширина 95% ДИ по событиям): "
                     f"{hw.min():.3f}–{hw.max():.3f} по {len(hw)} полным парам. Меньшие разности дизайн не "
                     "видит.\n")
        no_ev = sorted(pairwise.loc[pairwise["event_mean_diff"].isna(), "config"])
        if no_ev:
            why = ("нет кредитов конфигурации" if REFERENCE in set(ctx["credits"]["config"])
                   else "нет кредитов референса")
            L.append(f"Без сравнения по событиям ({why}): {', '.join(no_ev)}.\n")

    # ------------------------------------------------------- воспроизведение
    L.append("## Воспроизведение runs.csv\n")
    rep = cells.reproduction
    if rep.empty:
        L.append("Ни одной клетки со score — сверять нечего.\n")
    else:
        g = rep.groupby(["config", "source"]).agg(
            cells=("seed", "size"), max_abs_diff_padf=("abs_diff_padf", "max"),
            max_abs_diff_threshold=("abs_diff_threshold", "max"),
            max_abs_diff_alarms_per_hour=("abs_diff_alarms_per_hour", "max"),
            padf_match=("padf_match", "all"), threshold_match=("threshold_match", "all"),
            reproduced=("reproduced", "all"), expected_bit_identical=("expected_bit_identical", "all"),
        ).reset_index()
        L.append(_md(g, 6))
        L.append("\nДопуски: padf — 1e-9; порог — 1e-4 относительно max(1, |порог|). Пересчёт "
                 "`ctrl_untrained` НЕ обязан совпадать бит в бит: в исходном раннере случайные веса "
                 "инициализируются до `set_seed`, то есть зависят от состояния глобального генератора в "
                 "момент исходного прогона (здесь пересчёт детерминирован: `set_seed(сид)` перед сборкой "
                 "сети, — но это другие веса). Небольшие расхождения порога у бейзлайнов при пересчёте на "
                 "другой платформе (версии sklearn/numpy, Windows против Linux) ожидаемы; padf сверяется "
                 "отдельно. `reproduced` — совпали padf, порог и частота тревог (допуски те же; частота "
                 "тревог — 1e-9).\n")
        bad = rep[~rep["reproduced"]]
        if not bad.empty:
            L.append(f"Расхождения по клеткам ({len(bad)}): см. `reproduction.csv`.\n")
        off = g[~g["reproduced"]]
        if not off.empty:
            def why(r) -> str:
                return ("пересчитана с другими случайными весами необученной сети" if not r.expected_bit_identical
                        else "пересчитана" if r.source == "recomputed" else "сохранённые score")
            L.append("**score прогона не воспроизведены** в этом окружении: "
                     + ", ".join(f"`{r.config}` ({why(r)}; max |Δpadf| = {r.max_abs_diff_padf:.3f}, "
                                 f"max |Δпорог| = {r.max_abs_diff_threshold:.3f}, max |Δтревог/ч| = "
                                 f"{r.max_abs_diff_alarms_per_hour:.3f})" for r in off.itertuples())
                     + ". Числа разделов A, C и кредиты событий у этих клеток относятся к другим score "
                       "(другие веса или другая платформа), а не к score прогона, даже если padf совпал; "
                       "для них нужен `scores/` прогона.\n")

    # ------------------------------------------------------------ сверка ТЗ
    if ctx["tz_reference"]:
        c, band_rows, zero_v = _tz_computed(ctx)
        L.append("## ТЗ v2 §1 -> пересчёт\n")
        L.append("Опорные числа docs/TZ_PROTOCOL_V2.md §1 (пп. 2–5) для пилотного прогона `ft_aed_cv`. "
                 "«нет score» — для числа нужны score клеток, которых здесь нет (или есть не у всех "
                 "конфигураций); для строк по событиям — полные кредиты событий всех пар.\n")
        if ctx["unknown_datasets"]:
            L.append("**Диагностика по подмножеству фолдов**: числа ТЗ относятся ко всем фолдам прогона, "
                     "поэтому сверка ниже несопоставима и приведена только для проверки формата.\n")
        sv_flag = ctx["seedvar"]["zero_within_fold_variance"]
        n_ds = len(ctx["datasets"])
        L.append(_md(tz_reconciliation(
            c, band_rows, None if sv_flag.isna().all() else zero_v,
            computed_notes={"random_expected_padf": f"сидов: {ctx['random_seeds']}, фолдов: {n_ds}"}), 4))
        L.append("\nПояснения к расхождениям и определениям:\n")
        mc = block_ci_mc_range(runs, "base_iforest")
        if mc:
            _, lo_tz, lo_tol = TZ_REFERENCE["block_ci_lo.base_iforest"]
            _, hi_tz, hi_tol = TZ_REFERENCE["block_ci_hi.base_iforest"]
            L.append(f"- Блочный ДИ base_iforest − референс считается `report.compare_to_reference` "
                     "(5000 повторов, сид 0). Границы bootstrap — оценка Monte Carlo; опробованы сиды "
                     f"{_seeds_text(mc['seeds'])} при n_boot = {', '.join(map(str, mc['n_boots']))}. "
                     + _mc_bound_text("Нижняя", lo_tz, lo_tol, mc["lo_min"], mc["lo_max"]) + "; "
                     + _mc_bound_text("верхняя", hi_tz, hi_tol, mc["hi_min"], mc["hi_max"]) + ".")
        L.append("- «От 0 до 6.05 окон/ч» в ТЗ сверяется со средним по сидам конфигурации в фолде; "
                 f"максимум по отдельным клеткам — {runs['alarms_per_hour'].max():.2f}/ч.")
        col = "padf_ORACLE_1ph"
        if col in null.columns:
            L.append("- ORACLE 0.355 в ТЗ сверяется с клетками `ctrl_random` прогона (его сиды); среднее "
                     f"по всем {ctx['random_seeds']} сидам нулевого распределения — {null[col].mean():.3f}.")
        L.append("")

    # --------------------------------------------------------- чего не хватает
    L.append("## Чего не хватает\n")
    miss = cells.missing
    if miss.empty:
        L.append("Score есть у всех клеток `runs.csv`.\n")
    else:
        g = miss.groupby(["config", "reason"]).agg(
            folds=("dataset", lambda v: ", ".join(sorted(set(v)))),
            seeds=("seed", lambda v: _seeds_text(set(v))), cells=("seed", "size")).reset_index()
        L.append(_md(g, 0))
        events_note = (
            "Кредиты событий взяты из `events.csv` прогона, поэтому сравнение по событиям в F от score "
            "не зависит (полнота — по `event_complete` в `pairwise.csv`)."
            if ctx["credits_from_events_csv"] else
            "Нет и кредитов событий (нет `events.csv` прогона), поэтому нет сравнения по событиям в F "
            "(для всех пар, если среди них референс).")
        L.append("\nДля этих клеток отсутствуют: единицы тревог (окна, эпизоды, опора порога в "
                 "калибровке, раздел A), ORACLE-пересчёт (C), строки сверки ТЗ со статусом «нет score». "
                 f"{events_note} Числа по `runs.csv` (B — классификация, D — сопоставление, E, F — блоки и "
                 "фолды) посчитаны.\n")
        rates = " ".join(f"{r:g}" for r in ctx["oracle_rates"])
        cmd = (f"python scripts/diagnose_run.py --config {ctx['config_path'] or '<конфиг прогона>'} "
               f"--run-dir {ctx['run_dir']} --out {ctx['out']} --random-seeds {ctx['random_seeds']} "
               f"--n-boot {ctx['n_boot']} --oracle-rates {rates}"
               + (f" --datasets {' '.join(ctx['datasets'])}" if ctx["unknown_datasets"] else "")
               + (" --tz-reference" if ctx["tz_reference"] else ""))
        L.append(f"Повторить на машине, где есть `{ctx['run_dir']}/scores/` (score теста и `__calib`):\n")
        L.append(f"```bash\n{cmd}\n```\n")
    return "\n".join(L)


__all__ = [
    "DATASET_CHECK_KEYS", "MODEL_FREE_SCORES", "Protocol", "TZ_RANDOM_BAND", "TZ_REFERENCE",
    "TZ_ZERO_VARIANCE", "alarm_episode_starts", "alarm_units", "alarm_units_calib", "block_ci_mc_range",
    "cell_diagnostics", "classify_band", "cluster_bootstrap", "dataset_mismatches", "evaluate_model_free",
    "event_credits", "event_level_comparison",
    "fold_level_comparison", "model_free_scores", "oracle_report", "pairwise_table", "per_block_quantiles",
    "protocol_for_run", "random_band", "random_null", "run_diagnostics", "seed_variance",
    "stratified_resample", "summarize_alarm_units", "alarm_units_test", "tz_reconciliation",
]
