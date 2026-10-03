"""Event-level метрики — основная система оценки этого бенчмарка.

Почему точечные метрики здесь вторичны. Оператор дорожной службы не
реагирует на отдельные 30-секундные отсчёты: он реагирует на **событие**.
Одно ДТП, покрывающее 200 точек, — это одна единица учёта, а не двести.
Поточечный учёт порождает два противоположных искажения, и оба описаны
в литературе:

* **недооценка** без корректировки: модель, нашедшая событие с задержкой
  в один шаг, получает почти нулевой recall по точкам;
* **переоценка** с point-adjustment: сегмент целиком объявляется
  найденным по одной сработке, из-за чего случайный score превращается
  в SOTA (Kim et al., 2021). Количественный зазор между PA и event-level
  на одних и тех же моделях измерен в Zhong et al. (2026).

Поэтому основой служат три величины, напрямую отвечающие операционному
смыслу задачи:

``event_recall``
    доля событий, обнаруженных хотя бы раз при фиксированном FPR;
``median_delay_min``
    медианная задержка относительно времени официального отчёта
    (отрицательная = обнаружено раньше отчёта — цель);
``padf``
    затухающая награда за скорость (Gim et al., 2023): обнаружение через
    минуту и через час различаются принципиально, и метрика должна это
    отражать.

Порог всегда подбирается по **FPR на нормальных точках**, а не по
оптимальному F1 на тесте: подбор порога по тесту — скрытая утечка,
из-за которой числа в литературе несопоставимы (Alves et al., 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


#: Сколько подряд идущих окон должны превысить порог, чтобы тревога считалась
#: подтверждённой. Это не «сглаживание для красивых чисел», а воспроизведение
#: логики подтверждения из классических AID-алгоритмов (California): инцидент
#: порождает устойчивый сигнал, шум — одиночные выбросы.
#:
#: Без подтверждения метрика вырождается. Арифметика: на сети из 196 узлов при
#: FPR 1% и шаге 30 с случайный детектор выдаёт сотни одиночных сработок в час,
#: и хотя бы одна попадает в каждое событие — тогда event-recall случайного
#: score равен 1.0, и метрика становится бессмысленной. Это именно тот класс
#: дефектов, который Kim et al. (2021) описали для point-adjustment.
DEFAULT_PERSISTENCE: int = 3


def confirm_alarms(fired: np.ndarray, persistence: int = DEFAULT_PERSISTENCE) -> np.ndarray:
    """Оставить только тревоги, подтверждённые ``persistence`` окнами подряд.

    ``fired``: ``[n_windows, N]``, окна идут последовательно по времени.
    Возвращает маску той же формы: True в тех позициях, где у узла
    непрерывная серия превышений длиной не менее ``persistence``,
    причём отмечается **конец** серии — момент, когда тревога
    подтверждена и может быть выдана оператору.
    """
    if persistence <= 1:
        return fired
    out = fired.copy()
    for k in range(1, persistence):
        shifted = np.zeros_like(fired)
        shifted[k:] = fired[:-k]
        out &= shifted
    return out


def threshold_at_fpr(
    scores: np.ndarray,
    y: np.ndarray,
    fpr: float = 0.01,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> float:
    """Порог, дающий заданный FPR **подтверждённых** тревог на норме.

    Порог подбирается так, чтобы доля подтверждённых ложных тревог среди
    нормальных точек равнялась ``fpr``. Подбор ведётся по сырому порогу
    бинарным поиском, потому что подтверждение — нелинейная операция и
    квантиль по сырым score ей не соответствует.
    """
    if not 0 < fpr < 1:
        raise ValueError(f"fpr должен быть в (0,1), получено {fpr}")
    normal_mask = y == 0
    if not normal_mask.any():
        raise ValueError("нет нормальных точек — порог не определён")

    if persistence <= 1:
        return float(np.quantile(scores[normal_mask], 1.0 - fpr))

    n_normal = int(normal_mask.sum())
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return hi

    def fp_rate(thr: float) -> float:
        confirmed = confirm_alarms(scores > thr, persistence)
        return float((confirmed & normal_mask).sum() / n_normal)

    # монотонность по порогу гарантирует сходимость бинарного поиска
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fp_rate(mid) > fpr:
            lo = mid
        else:
            hi = mid
    return float(hi)


def threshold_at_alarm_rate(
    scores: np.ndarray,
    y: np.ndarray,
    target_per_hour: float,
    step_min: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> float:
    """Порог, дающий заданное число ложных тревог в час **на всю сеть**.

    Почему это, а не поточечный FPR — главная рабочая точка бенчмарка.

    Поточечный FPR не является операционной величиной и на сети из
    многих узлов вырождается. Арифметика: 30 узлов при шаге 1 мин — это
    1800 ячеек «узел × окно» в час, поэтому FPR = 1% означает 18 ложных
    тревог в час. Событие занимает десятки окон и несколько узлов, так
    что при таком бюджете тревог случайный score попадает внутрь почти
    любого события, и его event-recall приближается к единице. Метрика
    перестаёт различать модели — это тот же класс дефектов, который
    Kim et al. (2021) описали для point-adjustment, только в другом месте.

    Диспетчерская же оперирует не процентами, а числом тревог в час:
    бюджет внимания оператора конечен. Поэтому порог задаётся через него,
    и все модели сравниваются при **равном бюджете ложных тревог**.
    Поточечный FPR при этом тоже считается — для сопоставимости с
    литературой.
    """
    if target_per_hour <= 0:
        raise ValueError(f"target_per_hour должен быть > 0, получено {target_per_hour}")
    normal_mask = y == 0
    if not normal_mask.any():
        raise ValueError("нет нормальных точек — порог не определён")

    hours = scores.shape[0] * step_min / 60.0
    budget = target_per_hour * hours

    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return hi

    def n_false(thr: float) -> float:
        confirmed = confirm_alarms(scores > thr, persistence)
        return float((confirmed & normal_mask).sum())

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if n_false(mid) > budget:
            lo = mid
        else:
            hi = mid
    return float(hi)


def observed_fpr(
    scores: np.ndarray, y: np.ndarray, threshold: float, *, persistence: int = DEFAULT_PERSISTENCE
) -> float:
    """Фактический поточечный FPR при данном пороге — для сопоставимости."""
    confirmed = confirm_alarms(scores > threshold, persistence)
    normal = y == 0
    return float((confirmed & normal).sum() / max(1, int(normal.sum())))


def event_delays(
    scores: np.ndarray,
    event_id: np.ndarray,
    t_window: np.ndarray,
    events: pd.DataFrame,
    threshold: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> pd.DataFrame:
    """Для каждого события — первая подтверждённая сработка и задержка в минутах.

    Parameters
    ----------
    scores, event_id:
        ``[n_windows, N]``.
    t_window:
        ``[n_windows]`` время конца окна, минуты.
    events:
        реестр с ``event_id`` и ``t_report`` (минуты, та же шкала).

    Returns
    -------
    DataFrame с колонками ``event_id``, ``detected``, ``delay_min``,
    ``t_first_alarm``, ``n_alarms``.
    """
    fired = confirm_alarms(scores > threshold, persistence)
    rows = []
    for ev in events.itertuples():
        mask = (event_id == ev.event_id) & fired
        if mask.any():
            w_idx = np.where(mask.any(axis=1))[0]
            t_first = float(t_window[w_idx.min()])
            rows.append(
                {
                    "event_id": int(ev.event_id),
                    "detected": True,
                    "delay_min": t_first - float(ev.t_report),
                    "t_first_alarm": t_first,
                    "n_alarms": int(mask.sum()),
                }
            )
        else:
            rows.append(
                {
                    "event_id": int(ev.event_id),
                    "detected": False,
                    "delay_min": np.inf,
                    "t_first_alarm": np.nan,
                    "n_alarms": 0,
                }
            )
    return pd.DataFrame(rows)


def padf(delays: np.ndarray, *, half_life_min: float = 15.0) -> float:
    """Затухающая награда за раннее обнаружение (PAdf-подобно).

    Награда 1.0 — обнаружено не позже отчёта; далее экспоненциальное
    затухание с полупериодом ``half_life_min``; 0 — не обнаружено.
    Полупериод задаётся операционно (сколько стоит минута задержки) и
    обязательно фиксируется в конфиге, иначе метрика произвольна.
    """
    d = np.asarray(delays, dtype=float)
    credit = np.where(np.isfinite(d), 0.5 ** (np.maximum(d, 0.0) / half_life_min), 0.0)
    return float(credit.mean()) if credit.size else 0.0


def alarms_per_hour(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    step_min: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> float:
    """Подтверждённые ложные тревоги в час на всю сеть — операционная цена FPR.

    FPR 1% звучит безобидно, но на 196 узлах с шагом 30 с это десятки
    тревог в час. Без этой величины сравнение моделей оторвано от
    эксплуатации: диспетчер, получающий 40 ложных тревог в час,
    выключит систему независимо от её AUC.
    """
    confirmed = confirm_alarms(scores > threshold, persistence)
    false_alarms = int((confirmed & (y == 0)).sum())
    hours = scores.shape[0] * step_min / 60.0
    return float(false_alarms / hours) if hours > 0 else float("nan")


def event_level_report(
    scores: np.ndarray,
    y: np.ndarray,
    event_id: np.ndarray,
    t_window: np.ndarray,
    events: pd.DataFrame,
    *,
    alarm_budget_per_hour: float = 1.0,
    fpr: float | None = None,
    half_life_min: float = 15.0,
    step_min: float = 0.5,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, float]:
    """Полный event-level отчёт при равном бюджете ложных тревог.

    Рабочая точка задаётся числом подтверждённых ложных тревог в час на
    всю сеть (``alarm_budget_per_hour``) — это операционная величина, см.
    :func:`threshold_at_alarm_rate`. Если вместо неё задан ``fpr``,
    используется поточечный порог: режим оставлен для сопоставимости с
    литературой, но для ранжирования моделей не рекомендуется.
    """
    if len(events) == 0:
        raise ValueError("пустой реестр событий — метрики не определены")
    if fpr is not None:
        thr = threshold_at_fpr(scores, y, fpr, persistence=persistence)
    else:
        thr = threshold_at_alarm_rate(
            scores, y, alarm_budget_per_hour, step_min, persistence=persistence
        )
    det = event_delays(scores, event_id, t_window, events, thr, persistence=persistence)
    found = det.loc[det["detected"], "delay_min"].to_numpy()
    return {
        "threshold": thr,
        "alarm_budget_per_hour": alarm_budget_per_hour if fpr is None else float("nan"),
        "fpr_observed": observed_fpr(scores, y, thr, persistence=persistence),
        "persistence": persistence,
        "event_recall": float(det["detected"].mean()),
        "n_events": int(len(det)),
        "n_detected": int(det["detected"].sum()),
        "median_delay_min": float(np.median(found)) if found.size else float("nan"),
        "mean_delay_min": float(found.mean()) if found.size else float("nan"),
        "p90_delay_min": float(np.quantile(found, 0.9)) if found.size else float("nan"),
        "earlier_than_report_share": float((found < 0).mean()) if found.size else 0.0,
        "padf": padf(det["delay_min"].to_numpy(), half_life_min=half_life_min),
        "alarms_per_hour": alarms_per_hour(scores, y, thr, step_min, persistence=persistence),
    }


def operating_curve(
    scores: np.ndarray,
    y: np.ndarray,
    event_id: np.ndarray,
    t_window: np.ndarray,
    events: pd.DataFrame,
    *,
    budgets: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    half_life_min: float = 15.0,
    step_min: float = 0.5,
    persistence: int = DEFAULT_PERSISTENCE,
) -> pd.DataFrame:
    """Кривая «recall и задержка против бюджета ложных тревог в час».

    Одно число при одной рабочей точке скрывает форму компромисса: две
    модели с равным recall при 1 тревоге в час могут вести себя
    противоположно при 0.25, а именно там находится реальная рабочая
    точка диспетчерской — внимание оператора конечно.

    Ось в тревогах в час, а не в процентах FPR: это та единица, в
    которой решение принимает эксплуатант.
    """
    return pd.DataFrame(
        [
            event_level_report(
                scores, y, event_id, t_window, events,
                alarm_budget_per_hour=b, half_life_min=half_life_min,
                step_min=step_min, persistence=persistence,
            )
            for b in budgets
        ]
    )
