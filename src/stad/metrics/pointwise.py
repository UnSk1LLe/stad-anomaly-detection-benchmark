"""Поточечные метрики, включая демонстрацию инфляции point-adjustment.

Здесь сознательно реализован и сам протокол PA, хотя он непригоден для
отчётности. Причина: **инфляцию нужно показать, а не пересказать**.
Отдельная фигура бенчмарка сопоставляет PA-F1 и event-level F1 для всех
конфигураций, включая случайный score. Когда случайный детектор выходит
в верхнюю часть PA-рейтинга, становится видно, почему опубликованным
таблицам с PA-F1 доверять нельзя (Kim et al., 2021; Sehili et al., 2023).

Выбор основной поточечной метрики — **average precision (PR-AUC)**, не
ROC-AUC. Причина количественная: при экстремальном дисбалансе классов
случайный PR-AUC ограничен снизу базовой частотой аномалий, а ROC-AUC
нет, из-за чего ROC-семейство завышается при best-of-N отчётности уже
при N≈9–11, тогда как PR-метрики остаются плоскими при любом N
(Lyu, 2026).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """PR-AUC по точкам. Основная поточечная метрика бенчмарка."""
    yy = y.ravel()
    if yy.min() == yy.max():
        return float("nan")
    return float(average_precision_score(yy, scores.ravel()))


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC. Репортится **только** для сопоставимости с литературой.

    Не используется для ранжирования моделей: при редких аномалиях он
    оптимистичен и подвержен инфляции при best-of-N (Lyu, 2026).
    """
    yy = y.ravel()
    if yy.min() == yy.max():
        return float("nan")
    return float(roc_auc_score(yy, scores.ravel()))


def best_f1(scores: np.ndarray, y: np.ndarray, *, n_thresholds: int = 200) -> tuple[float, float]:
    """Максимальный поточечный F1 по сетке порогов и сам порог.

    Это «оракульный» F1: порог подобран по тесту. Репортится как **верхняя
    граница**, а не как достижимое качество, и всегда рядом с F1 при
    пороге, выставленном по FPR на нормальных данных.
    """
    s, yy = scores.ravel(), y.ravel()
    lo, hi = np.quantile(s, 0.5), s.max()
    best, best_thr = 0.0, float(hi)
    for thr in np.linspace(lo, hi, n_thresholds):
        pred = s > thr
        tp = float((pred & (yy == 1)).sum())
        fp = float((pred & (yy == 0)).sum())
        fn = float((~pred & (yy == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best:
            best, best_thr = f1, float(thr)
    return best, best_thr


def f1_at_threshold(scores: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Честный поточечный F1 при пороге, заданном извне (по FPR)."""
    pred = scores.ravel() > threshold
    yy = y.ravel()
    tp = float((pred & (yy == 1)).sum())
    fp = float((pred & (yy == 0)).sum())
    fn = float((~pred & (yy == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0


def point_adjust(pred: np.ndarray, y: np.ndarray, event_id: np.ndarray) -> np.ndarray:
    """Применить протокол point-adjustment: сегмент целиком = True по одной сработке.

    Реализовано **исключительно для демонстрации инфляции**. Любое число,
    полученное после этой функции, в отчёте помечается как недопустимое
    для сравнения моделей.
    """
    adjusted = pred.copy()
    for eid in np.unique(event_id[event_id >= 0]):
        seg = event_id == eid
        if (pred & seg).any():
            adjusted[seg] = True
    return adjusted


def pa_f1(scores: np.ndarray, y: np.ndarray, event_id: np.ndarray, *, n_thresholds: int = 200) -> float:
    """F1 после point-adjustment с оптимальным порогом — метрика-антипример.

    Ожидаемое поведение в отчёте: даже случайный score получает здесь
    высокое значение. Это и есть воспроизведение результата Kim et al.
    на собственных данных — элемент валидации протокола, а не оценка
    качества модели.
    """
    s, yy = scores.ravel(), y.ravel()
    eid = event_id.ravel()
    lo, hi = np.quantile(s, 0.5), s.max()
    best = 0.0
    for thr in np.linspace(lo, hi, n_thresholds):
        pred = point_adjust(s > thr, yy, eid)
        tp = float((pred & (yy == 1)).sum())
        fp = float((pred & (yy == 0)).sum())
        fn = float((~pred & (yy == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        best = max(best, f1)
    return best


def pointwise_report(
    scores: np.ndarray,
    y: np.ndarray,
    event_id: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Все поточечные величины сразу, с явной пометкой назначения каждой."""
    out = {
        "average_precision": average_precision(scores, y),
        "prevalence": float(y.mean()),
        "roc_auc_reference_only": roc_auc(scores, y),
        "pa_f1_INVALID_for_ranking": pa_f1(scores, y, event_id),
    }
    oracle_f1, oracle_thr = best_f1(scores, y)
    out["f1_oracle_threshold_upper_bound"] = oracle_f1
    out["oracle_threshold"] = oracle_thr
    if threshold is not None:
        out["f1_at_fpr_threshold"] = f1_at_threshold(scores, y, threshold)
    # во сколько раз PA завышает F1 относительно честного
    honest = out.get("f1_at_fpr_threshold", oracle_f1)
    out["pa_inflation_ratio"] = (
        out["pa_f1_INVALID_for_ranking"] / honest if honest > 1e-9 else float("inf")
    )
    # нормированная PR: во сколько раз лучше случайного
    out["ap_lift_over_random"] = (
        out["average_precision"] / out["prevalence"] if out["prevalence"] > 0 else float("nan")
    )
    return out
