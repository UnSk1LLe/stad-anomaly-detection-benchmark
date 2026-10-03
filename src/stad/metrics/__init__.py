"""Метрики бенчмарка.

Иерархия назначения строго фиксирована, чтобы в отчёте нельзя было
случайно сравнить модели по инфлированной величине:

* **Основные (ранжирование моделей):** ``event_recall``,
  ``median_delay_min``, ``padf``, ``average_precision``.
* **Операционные (выбор рабочей точки):** ``alarms_per_hour``,
  ``operating_curve``.
* **Справочные (сопоставимость с литературой):** ``roc_auc``,
  ``f1_oracle_threshold_upper_bound``.
* **Антипримеры (демонстрация инфляции, НЕ для ранжирования):**
  ``pa_f1``, ``pa_inflation_ratio``.
"""
from .event_level import (
    DEFAULT_PERSISTENCE,
    alarms_per_hour,
    confirm_alarms,
    event_delays,
    event_level_report,
    observed_fpr,
    operating_curve,
    padf,
    threshold_at_alarm_rate,
    threshold_at_fpr,
)
from .pointwise import (
    average_precision,
    best_f1,
    f1_at_threshold,
    pa_f1,
    point_adjust,
    pointwise_report,
    roc_auc,
)
from .stats import (
    bootstrap_ci,
    friedman,
    mean_ranks,
    nemenyi_cd,
    paired_bootstrap_test,
    rank_matrix,
    variance_decomposition,
)

#: Метрики, по которым разрешено ранжировать модели, и направление «лучше».
PRIMARY_METRICS: dict[str, bool] = {
    "event_recall": True,
    "padf": True,
    "average_precision": True,
    "median_delay_min": False,      # меньше = лучше (раньше обнаружено)
}

#: Метрики, запрещённые для ранжирования (попадают в отчёт с пометкой).
FORBIDDEN_FOR_RANKING: tuple[str, ...] = (
    "pa_f1_INVALID_for_ranking",
    "f1_oracle_threshold_upper_bound",
    "roc_auc_reference_only",
)

__all__ = [
    "threshold_at_fpr", "event_delays", "padf", "alarms_per_hour",
    "confirm_alarms", "DEFAULT_PERSISTENCE", "threshold_at_alarm_rate", "observed_fpr",
    "event_level_report", "operating_curve",
    "average_precision", "roc_auc", "best_f1", "f1_at_threshold",
    "point_adjust", "pa_f1", "pointwise_report",
    "friedman", "nemenyi_cd", "mean_ranks", "rank_matrix",
    "bootstrap_ci", "paired_bootstrap_test", "variance_decomposition",
    "PRIMARY_METRICS", "FORBIDDEN_FOR_RANKING",
]


def full_report(
    scores,
    data,
    *,
    alarm_budget_per_hour: float = 1.0,
    half_life_min: float = 15.0,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, float]:
    """Единая точка входа: event-level + поточечные метрики для одного прогона."""
    step_min = float(data.meta.get("step_min", 0.5))
    ev = event_level_report(
        scores, data.y_test, data.event_id_test, data.t_test, data.events,
        alarm_budget_per_hour=alarm_budget_per_hour,
        half_life_min=half_life_min, step_min=step_min,
        persistence=persistence,
    )
    pt = pointwise_report(scores, data.y_test, data.event_id_test, threshold=ev["threshold"])
    return {**ev, **pt}
