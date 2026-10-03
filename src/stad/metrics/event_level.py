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

from dataclasses import dataclass

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


def window_spacing(t: np.ndarray) -> float:
    """Типичный шаг между соседними окнами, минуты (медиана положительных разностей)."""
    d = np.diff(np.asarray(t, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def contiguous_segments(t: np.ndarray, spacing: float | None = None) -> np.ndarray:
    """Номера непрерывных участков по времени окон: ``[n]`` целых.

    Окна соседние в массиве, но не во времени, если между ними вырезан
    кусок (очистка вокруг событий) или лежит ночной разрыв между днями.
    Подтверждение «три окна подряд» через такой разрыв склеило бы два
    не связанных друг с другом отрезка, поэтому там сегмент обрывается.
    """
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    sp = float(spacing) if spacing else window_spacing(t)
    brk = np.diff(t) > 1.5 * sp
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)


def confirm_alarms(
    fired: np.ndarray,
    persistence: int = DEFAULT_PERSISTENCE,
    *,
    segments: np.ndarray | None = None,
) -> np.ndarray:
    """Оставить только тревоги, подтверждённые ``persistence`` окнами подряд.

    ``fired``: ``[n_windows, N]``, окна идут последовательно по времени.
    Возвращает маску той же формы: True в тех позициях, где у узла
    непрерывная серия превышений длиной не менее ``persistence``,
    причём отмечается **конец** серии — момент, когда тревога
    подтверждена и может быть выдана оператору.

    ``segments`` (``[n_windows]``, см. :func:`contiguous_segments`): серия
    не может пересекать границу сегмента. Без него окна считаются
    сплошными — верно для теста внутри одного дня, неверно для валидации
    с вырезанными окнами.
    """
    if persistence <= 1:
        return fired
    out = fired.copy()
    for k in range(1, persistence):
        shifted = np.zeros_like(fired)
        shifted[k:] = fired[:-k]
        if segments is not None:
            same = np.zeros(len(fired), dtype=bool)
            same[k:] = segments[k:] == segments[:-k]
            shifted &= same.reshape((-1,) + (1,) * (fired.ndim - 1))
        out &= shifted
    return out


@dataclass(frozen=True)
class Calibration:
    """Нормальные окна, по которым подбирается порог. МЕТОК ТЕСТА ЗДЕСЬ НЕТ.

    Порог, подобранный по нормальным точкам самого теста, использует
    тестовые метки: чтобы знать, какие точки нормальные, надо заглянуть в
    ``y_test``. Поэтому порог калибруется на валидационных днях, где
    аномальных окон по построению нет, и переносится на тест без
    изменений.

    ``scores`` — score валидации **уже сведённые** тем же способом, что и
    тестовые (``evaluation_view``), ``[n_val, M]``.
    """

    scores: np.ndarray
    step_min: float
    segments: np.ndarray | None = None

    @property
    def hours(self) -> float:
        return self.scores.shape[0] * self.step_min / 60.0


def threshold_from_calibration(
    calib: Calibration,
    target_per_hour: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
) -> float:
    """Порог на ``target_per_hour`` подтверждённых ложных тревог в час по валидации.

    Сигнатура не принимает ни меток, ни тестовых score: зависимость порога
    от теста невозможна по построению (регрессионный тест
    ``test_threshold_independent_of_test_labels``).
    """
    return threshold_at_alarm_rate(
        calib.scores, np.zeros(calib.scores.shape, dtype=int), target_per_hour, calib.step_min,
        persistence=persistence, segments=calib.segments,
    )


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
    segments: np.ndarray | None = None,
) -> float:
    """Порог, дающий заданное число ложных тревог в час **на всю сеть**.

    ``y`` определяет, какие точки считаются нормальными. Для основного
    протокола сюда передаются валидационные окна и нули
    (:func:`threshold_from_calibration`); вызов с ``y`` теста подбирает
    порог по тестовым меткам и допустим только как оракул.

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
        confirmed = confirm_alarms(scores > thr, persistence, segments=segments)
        return float((confirmed & normal_mask).sum())

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if n_false(mid) > budget:
            lo = mid
        else:
            hi = mid
    return float(hi)


def observed_fpr(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    *,
    persistence: int = DEFAULT_PERSISTENCE,
    segments: np.ndarray | None = None,
) -> float:
    """Фактический поточечный FPR при данном пороге — для сопоставимости."""
    confirmed = confirm_alarms(scores > threshold, persistence, segments=segments)
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
    segments: np.ndarray | None = None,
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
    fired = confirm_alarms(scores > threshold, persistence, segments=segments)
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
    segments: np.ndarray | None = None,
) -> float:
    """Подтверждённые ложные тревоги в час на всю сеть — операционная цена FPR.

    FPR 1% звучит безобидно, но на 196 узлах с шагом 30 с это десятки
    тревог в час. Без этой величины сравнение моделей оторвано от
    эксплуатации: диспетчер, получающий 40 ложных тревог в час,
    выключит систему независимо от её AUC.
    """
    confirmed = confirm_alarms(scores > threshold, persistence, segments=segments)
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
    calibration: Calibration | None = None,
    segments: np.ndarray | None = None,
) -> dict[str, float]:
    """Полный event-level отчёт при равном бюджете ложных тревог.

    Рабочая точка задаётся числом подтверждённых ложных тревог в час на
    всю сеть (``alarm_budget_per_hour``) — это операционная величина, см.
    :func:`threshold_at_alarm_rate`.

    **Откуда берётся порог.** С ``calibration`` — по нормальным окнам
    валидации, тест порога не видит (основной протокол, ``threshold_source
    = "validation"``). Без неё порог подбирается по нормальным точкам
    теста, то есть по тестовым меткам: это оракульный режим для
    юнит-тестов и диагностики, а не результат (``threshold_source =
    "test_normals_ORACLE"``). Пайплайн (:func:`stad.metrics.full_report`)
    без калибровки не работает вовсе.

    Если вместо бюджета задан ``fpr``, используется поточечный порог:
    режим для сопоставимости с литературой, всегда оракульный.
    """
    if len(events) == 0:
        raise ValueError("пустой реестр событий — метрики не определены")
    if fpr is not None:
        thr = threshold_at_fpr(scores, y, fpr, persistence=persistence)
        source = "test_normals_ORACLE"
    elif calibration is not None:
        thr = threshold_from_calibration(calibration, alarm_budget_per_hour, persistence=persistence)
        source = "validation"
    else:
        thr = threshold_at_alarm_rate(
            scores, y, alarm_budget_per_hour, step_min, persistence=persistence,
            segments=segments,
        )
        source = "test_normals_ORACLE"
    det = event_delays(scores, event_id, t_window, events, thr, persistence=persistence,
                       segments=segments)
    found = det.loc[det["detected"], "delay_min"].to_numpy()
    return {
        "threshold": thr,
        "threshold_source": source,
        "alarm_budget_per_hour": alarm_budget_per_hour if fpr is None else float("nan"),
        "fpr_observed": observed_fpr(scores, y, thr, persistence=persistence, segments=segments),
        "persistence": persistence,
        "event_recall": float(det["detected"].mean()),
        "n_events": int(len(det)),
        "n_detected": int(det["detected"].sum()),
        "median_delay_min": float(np.median(found)) if found.size else float("nan"),
        "mean_delay_min": float(found.mean()) if found.size else float("nan"),
        "p90_delay_min": float(np.quantile(found, 0.9)) if found.size else float("nan"),
        "earlier_than_report_share": float((found < 0).mean()) if found.size else 0.0,
        "padf": padf(det["delay_min"].to_numpy(), half_life_min=half_life_min),
        "alarms_per_hour": alarms_per_hour(scores, y, thr, step_min, persistence=persistence,
                                           segments=segments),
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
    calibration: Calibration | None = None,
    segments: np.ndarray | None = None,
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
                calibration=calibration, segments=segments,
            )
            for b in budgets
        ]
    )
