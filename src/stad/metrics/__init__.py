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
import re

import numpy as np

from .event_level import (
    DEFAULT_PERSISTENCE,
    Calibration,
    alarms_per_hour,
    confirm_alarms,
    contiguous_segments,
    event_delays,
    event_level_report,
    observed_fpr,
    operating_curve,
    padf,
    threshold_at_alarm_rate,
    threshold_at_fpr,
    threshold_from_calibration,
    window_spacing,
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
    "Calibration", "threshold_from_calibration", "contiguous_segments", "window_spacing",
    "reduce_scores", "make_calibration", "eval_segments", "NODE_REDUCERS",
    "average_precision", "roc_auc", "best_f1", "f1_at_threshold",
    "point_adjust", "pa_f1", "pointwise_report",
    "friedman", "nemenyi_cd", "mean_ranks", "rank_matrix",
    "bootstrap_ci", "paired_bootstrap_test", "variance_decomposition",
    "PRIMARY_METRICS", "FORBIDDEN_FOR_RANKING", "evaluation_view",
]


#: Способы свести score по оси узлов. ``q99``/``q95`` — квантили по узлам;
#: ``max`` — статистика с тяжёлым хвостом, а квантиль устойчивее к одиночному шумному узлу.
NODE_REDUCERS: tuple[str, ...] = ("max", "q99", "q95", "mean")


def _node_reducer(reduce: str):
    if reduce == "max":
        return lambda s: np.max(s, axis=1, keepdims=True)
    if reduce == "mean":
        return lambda s: np.mean(s, axis=1, keepdims=True)
    m = re.fullmatch(r"q(\d{1,2})", reduce)
    if m and 0 < int(m.group(1)) < 100:
        q = int(m.group(1)) / 100.0
        return lambda s: np.quantile(s, q, axis=1, keepdims=True)
    raise ValueError(f"неизвестная агрегация по узлам {reduce!r}; доступны: {NODE_REDUCERS}")


def reduce_scores(scores, data, *, reduce: str = "max"):
    """Свести score к гранулярности меток: тест и калибровка должны проходить через одно и то же.

    Для корридор-уровневых меток (FT-AED) — агрегация по узлам, иначе без изменений.
    """
    if not data.meta.get("labels_are_corridor_level"):
        _node_reducer(reduce)           # проверить имя, даже если не применяется
        return scores
    return _node_reducer(reduce)(scores)


def eval_segments(data) -> np.ndarray:
    """Непрерывные участки тестовых окон: подтверждение не пересекает границу дней."""
    return contiguous_segments(data.t_test)


def make_calibration(calib_scores, data, *, reduce: str = "max") -> Calibration:
    """Калибровочная выборка порога: score окон ``data.X_calib`` (дни валидации).

    Выборка не сплошная (размеченные окна вырезаны), поэтому нужен
    ``data.t_calib``: подтверждение через разрыв дало бы ложные серии. Метки
    теста в калибровке не участвуют. Если выборки нет, порог не калибруется
    вовсе — подставлять тест или train нельзя: первое утечка, второе
    оптимистично смещено.
    """
    if calib_scores is None:
        raise ValueError(
            "нет score калибровочной выборки: порог калибруется на окнах дней валидации, "
            "а не на тесте (CLAUDE.md, правило 3)"
        )
    if getattr(data, "t_calib", None) is None:
        raise ValueError(
            "SplitData.t_calib не задан — калибровочные окна не привязаны ко времени, "
            "калибровка порога невозможна"
        )
    calib_scores = np.asarray(calib_scores)
    if len(calib_scores) != len(data.t_calib):
        raise ValueError(
            f"score калибровки {calib_scores.shape} не соответствует t_calib {data.t_calib.shape}"
        )
    spacing = window_spacing(data.t_test)
    return Calibration(
        scores=reduce_scores(calib_scores, data, reduce=reduce),
        step_min=float(data.meta.get("step_min", 0.5)),
        segments=contiguous_segments(data.t_calib, spacing),
    )


def evaluation_view(scores, data, *, reduce: str = "max"):
    """Привести score и метки к той гранулярности, на которой заданы метки.

    В FT-AED метки **корридор-уровневые**: событие помечено сразу на всех
    196 узлах, пространственной локализации нет. Оценивать такой датасет
    поузловым решением неправильно по двум причинам:

    * у каждого события появляется 196 независимых шансов быть задетым
      шумом, и event-recall случайного score растёт искусственно;
    * модель, которая корректно зажигает много узлов выше по потоку
      (именно так и выглядит затор от ДТП), штрафуется как источник
      множества ложных тревог, тогда как это **одна** тревога.

    Поэтому score сводится по оси узлов (по умолчанию максимумом —
    инцидент локален, и сеть должна срабатывать по сильнейшему узлу), а
    метки берутся как есть: они одинаковы для всех узлов.

    Для датасетов с поузловыми метками (синтетический коридор) функция
    возвращает входные массивы без изменений.
    """
    if not data.meta.get("labels_are_corridor_level"):
        _node_reducer(reduce)
        return scores, data.y_test, data.event_id_test
    return (
        reduce_scores(scores, data, reduce=reduce),
        data.y_test[:, :1],
        data.event_id_test[:, :1],
    )


def full_report(
    scores,
    data,
    *,
    calib_scores,
    alarm_budget_per_hour: float = 1.0,
    half_life_min: float = 15.0,
    persistence: int = DEFAULT_PERSISTENCE,
    reduce: str = "max",
) -> dict[str, float]:
    """Единая точка входа: event-level + поточечные метрики для одного прогона.

    ``calib_scores`` обязателен: порог калибруется по окнам дней
    валидации и переносится на тест без изменений. Подбор порога по
    тестовым меткам запрещён (CLAUDE.md, правило 3).
    """
    step_min = float(data.meta.get("step_min", 0.5))
    s, y, eid = evaluation_view(scores, data, reduce=reduce)
    ev = event_level_report(
        s, y, eid, data.t_test, data.events,
        alarm_budget_per_hour=alarm_budget_per_hour,
        half_life_min=half_life_min, step_min=step_min,
        persistence=persistence,
        calibration=make_calibration(calib_scores, data, reduce=reduce),
        segments=eval_segments(data),
    )
    pt = pointwise_report(s, y, eid, threshold=ev["threshold"])
    return {**ev, **pt}
