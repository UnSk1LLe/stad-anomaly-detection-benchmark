"""Тесты метрик. Самый важный файл в репозитории.

Если сломаны метрики, все остальные результаты — шум, который выглядит
как наука. Поэтому здесь проверяются не «работает ли код», а
**предсказания литературы**: случайный score должен провалиться на
честных метриках и преуспеть на point-adjustment. Если этот тест
перестанет проходить, значит протокол перестал быть защищённым.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stad.metrics.event_level import (
    alarms_per_hour,
    event_delays,
    event_level_report,
    operating_curve,
    padf,
    threshold_at_fpr,
)
from stad.metrics.pointwise import average_precision, pa_f1, point_adjust, pointwise_report


@pytest.fixture
def toy():
    """Маленький синтетический тест: 200 окон × 4 узла, 3 события."""
    rng = np.random.default_rng(0)
    n, N = 200, 4
    y = np.zeros((n, N), dtype=int)
    eid = np.full((n, N), -1, dtype=int)
    events = []
    for k, start in enumerate((40, 100, 160)):
        y[start : start + 8, 1:3] = 1
        eid[start : start + 8, 1:3] = k
        events.append({"event_id": k, "t_report": float(start + 2)})
    t = np.arange(n, dtype=float)
    return y, eid, t, pd.DataFrame(events), rng


def test_random_score_ap_near_prevalence(toy):
    """Случайный score: AP ≈ доля аномалий (Lyu, 2026: пол PR-метрики)."""
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    ap = average_precision(scores, y)
    prevalence = y.mean()
    assert ap < prevalence * 2.0, f"AP={ap:.4f} слишком высок при prevalence={prevalence:.4f}"


def test_random_score_padf_low(toy):
    """Случайный score: padf близок к нулю при FPR 1%."""
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    rep = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0)
    assert rep["padf"] < 0.4, f"padf={rep['padf']:.3f} — метрика обманываема случайным score"


def test_point_adjustment_inflates_random_score(toy):
    """Воспроизведение Kim et al. (2021): PA поднимает случайный score.

    Это тест на **наличие** инфляции: если он перестанет проходить,
    значит реализация PA неверна, и зазор «PA против event-level»
    нельзя предъявлять как аргумент в диссертации.
    """
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    honest = average_precision(scores, y)
    inflated = pa_f1(scores, y, eid)
    assert inflated > honest * 3, (
        f"PA-F1={inflated:.3f} не превышает честную метрику={honest:.4f} — "
        "проверьте реализацию point_adjust"
    )
    assert inflated > 0.3, f"PA-F1={inflated:.3f}: ожидается заметная инфляция"


def test_point_adjust_marks_whole_segment():
    """Одна сработка внутри сегмента помечает весь сегмент."""
    pred = np.array([False, False, True, False, False, False])
    y = np.array([0, 1, 1, 1, 0, 0])
    eid = np.array([-1, 0, 0, 0, -1, -1])
    adj = point_adjust(pred, y, eid)
    assert adj.tolist() == [False, True, True, True, False, False]


def test_oracle_score_is_perfect(toy):
    """Идеальный score должен давать recall 1.0 и отрицательную задержку."""
    y, eid, t, events, _ = toy
    scores = y.astype(float) + 1e-6
    rep = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0)
    assert rep["event_recall"] == 1.0
    assert rep["padf"] > 0.6
    # событие помечено с t_report+(-2), значит первая сработка раньше отчёта
    assert rep["median_delay_min"] <= 0.0


def test_constant_score_detects_nothing(toy):
    """Константа: порог по FPR отсекает всё, recall падает к нулю."""
    y, eid, t, events, _ = toy
    scores = np.zeros_like(y, dtype=float)
    rep = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0)
    assert rep["event_recall"] == 0.0
    assert rep["padf"] == 0.0


def test_threshold_matches_requested_fpr(toy):
    """Порог даёт заданный FPR подтверждённых тревог на нормальных точках.

    Проверяется в двух режимах: без подтверждения (сырой квантиль) и с
    подтверждением (бинарный поиск по нелинейной функции).
    """
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    for fpr in (0.01, 0.05):
        thr = threshold_at_fpr(scores, y, fpr, persistence=1)
        actual = ((scores > thr) & (y == 0)).sum() / (y == 0).sum()
        assert abs(actual - fpr) < 0.01, f"persistence=1: цель {fpr}, факт {actual:.4f}"

    from stad.metrics.event_level import confirm_alarms

    for fpr in (0.01, 0.05):
        thr = threshold_at_fpr(scores, y, fpr, persistence=3)
        confirmed = confirm_alarms(scores > thr, 3)
        actual = (confirmed & (y == 0)).sum() / (y == 0).sum()
        assert actual <= fpr + 0.005, f"persistence=3: цель {fpr}, факт {actual:.4f}"


def test_padf_rewards_earlier_detection():
    """Чем раньше обнаружено, тем выше награда; не обнаружено = 0."""
    early = padf(np.array([-5.0, -5.0]), half_life_min=15.0)
    late = padf(np.array([30.0, 30.0]), half_life_min=15.0)
    never = padf(np.array([np.inf, np.inf]), half_life_min=15.0)
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(0.25, abs=1e-6)
    assert never == 0.0
    assert early > late > never


def test_delay_sign_convention(toy):
    """Отрицательная задержка = обнаружено раньше отчёта.

    Одиночная сработка, поэтому подтверждение отключено: здесь
    проверяется знак, а не логика подтверждения.
    """
    y, eid, t, events, _ = toy
    scores = np.zeros_like(y, dtype=float)
    scores[40, 1] = 100.0            # сработка при t=40, отчёт при t=42
    det = event_delays(scores, eid, t, events, threshold=1.0, persistence=1)
    row = det[det["event_id"] == 0].iloc[0]
    assert row["detected"]
    assert row["delay_min"] == pytest.approx(-2.0)


def test_operating_curve_monotone_recall(toy):
    """При росте бюджета ложных тревог recall не убывает."""
    y, eid, t, events, rng = toy
    scores = y * 2.0 + rng.standard_normal(y.shape) * 0.5
    curve = operating_curve(scores, y, eid, t, events, step_min=1.0).sort_values(
        "alarm_budget_per_hour"
    )
    recalls = curve["event_recall"].to_numpy()
    assert np.all(np.diff(recalls) >= -1e-9), f"recall не монотонен: {recalls}"


def test_alarms_per_hour_scales_with_threshold(toy):
    """Операционная цена: ниже порог — больше тревог в час."""
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    strict = alarms_per_hour(scores, y, threshold_at_fpr(scores, y, 0.001), step_min=1.0)
    loose = alarms_per_hour(scores, y, threshold_at_fpr(scores, y, 0.05), step_min=1.0)
    assert loose > strict


def test_pointwise_report_flags_forbidden_keys(toy):
    """Имена запрещённых метрик содержат предупреждение прямо в ключе."""
    y, eid, t, events, rng = toy
    rep = pointwise_report(rng.standard_normal(y.shape), y, eid)
    assert "pa_f1_INVALID_for_ranking" in rep
    assert "roc_auc_reference_only" in rep
    assert rep["pa_inflation_ratio"] > 1.0


def test_confirmation_suppresses_isolated_alarms():
    """Подтверждение гасит одиночные выбросы, но пропускает устойчивый сигнал.

    Логика California: инцидент даёт серию превышений подряд, шум — нет.
    Без этого правила event-recall случайного детектора вырождается в 1.0
    на сети из сотен узлов (см. DEFAULT_PERSISTENCE).
    """
    from stad.metrics.event_level import confirm_alarms

    fired = np.array([[True], [False], [True], [True], [True], [False]])
    out = confirm_alarms(fired, persistence=3)
    assert out.ravel().tolist() == [False, False, False, False, True, False]
    assert confirm_alarms(fired, persistence=1).ravel().tolist() == fired.ravel().tolist()


def test_persistence_kills_random_event_recall(toy):
    """С подтверждением случайный score теряет event-recall.

    Это регрессионный тест на главный дефект, который выявил первый
    smoke-прогон: без подтверждения случайный детектор обнаруживал
    100% событий при FPR 1%.
    """
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    naive = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0, persistence=1)
    strict = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0, persistence=3)
    assert strict["event_recall"] <= naive["event_recall"]
    assert strict["padf"] <= naive["padf"]


def test_persistence_keeps_oracle_recall(toy):
    """Подтверждение не должно ломать идеальный детектор."""
    y, eid, t, events, _ = toy
    scores = y.astype(float) * 10.0
    rep = event_level_report(scores, y, eid, t, events, fpr=0.01, step_min=1.0, persistence=3)
    assert rep["event_recall"] == 1.0, "подтверждение съело настоящие события"


def test_alarm_budget_is_respected(toy):
    """Порог по бюджету действительно укладывается в заданное число тревог в час.

    Это и есть определение рабочей точки бенчмарка: все модели
    сравниваются при равном бюджете внимания оператора.
    """
    from stad.metrics.event_level import threshold_at_alarm_rate, alarms_per_hour

    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    for budget in (0.5, 1.0, 5.0):
        thr = threshold_at_alarm_rate(scores, y, budget, step_min=1.0, persistence=3)
        actual = alarms_per_hour(scores, y, thr, step_min=1.0, persistence=3)
        assert actual <= budget * 1.3 + 0.2, f"бюджет {budget}, факт {actual:.2f}"


def test_random_loses_at_operational_budget(toy):
    """ГЛАВНАЯ проверка протокола: при операционном бюджете случайный score проигрывает.

    Регрессионный тест на дефект, выявленный smoke-прогоном: при
    поточечном FPR = 1% случайный детектор обнаруживал почти все
    события и обходил PCA. При бюджете в тревогах в час этого
    происходить не должно.
    """
    y, eid, t, events, rng = toy
    random_scores = rng.standard_normal(y.shape)
    informed = y * 2.0 + rng.standard_normal(y.shape) * 0.5

    r = event_level_report(random_scores, y, eid, t, events,
                           alarm_budget_per_hour=1.0, step_min=1.0)
    i = event_level_report(informed, y, eid, t, events,
                           alarm_budget_per_hour=1.0, step_min=1.0)
    assert i["padf"] > r["padf"], (
        f"информированный score (padf={i['padf']:.3f}) не обходит случайный "
        f"(padf={r['padf']:.3f}) — метрика обманываема"
    )
    assert r["event_recall"] < 0.7, f"случайный event_recall={r['event_recall']:.2f} слишком высок"


# ------------------------------------------------ порог калибруется на валидации
def _calibration(rng, n=300, N=4, step=1.0):
    from stad.metrics import Calibration

    return Calibration(scores=rng.standard_normal((n, N)), step_min=step)


def test_threshold_independent_of_test_labels(toy):
    """КРИТИЧНО: порог не зависит от тестовых меток и тестовых score.

    Регрессионный тест на утечку: раньше порог подбирался по нормальным
    точкам теста, то есть по ``y_test``. Здесь метки теста заменяются
    целиком — порог при калибровке по валидации обязан остаться тем же,
    а оракульный режим (по тесту) обязан измениться: тест различает оба.
    """
    y, eid, t, events, rng = toy
    scores = rng.standard_normal(y.shape)
    calib = _calibration(rng)

    base = event_level_report(scores, y, eid, t, events, calibration=calib, step_min=1.0)
    y_alt = 1 - y                                                # метки теста инвертированы
    alt = event_level_report(scores, y_alt, eid, t, events, calibration=calib, step_min=1.0)
    y_perm = rng.permutation(y.ravel()).reshape(y.shape)         # и перемешаны
    perm = event_level_report(scores, y_perm, eid, t, events, calibration=calib, step_min=1.0)
    other_scores = event_level_report(scores * 3.0 + 1.0, y, eid, t, events,
                                      calibration=calib, step_min=1.0)

    assert base["threshold"] == alt["threshold"] == perm["threshold"] == other_scores["threshold"]
    assert base["threshold_source"] == "validation"

    leaky = event_level_report(scores, y, eid, t, events, step_min=1.0)
    leaky_alt = event_level_report(scores, y_alt, eid, t, events, step_min=1.0)
    assert leaky["threshold_source"] == "test_normals_ORACLE"
    assert leaky["threshold"] != leaky_alt["threshold"], (
        "оракульный режим должен зависеть от меток теста — иначе тест выше ничего не различает"
    )


def test_calibration_threshold_meets_budget_on_validation(toy):
    """Порог по валидации укладывается в бюджет тревог на самой валидации."""
    from stad.metrics import threshold_from_calibration

    _, _, _, _, rng = toy
    calib = _calibration(rng, n=2000)
    for budget in (0.5, 2.0, 6.0):
        thr = threshold_from_calibration(calib, budget, persistence=3)
        actual = alarms_per_hour(calib.scores, np.zeros(calib.scores.shape, dtype=int), thr,
                                 1.0, persistence=3)
        assert actual <= budget * 1.3 + 0.2, f"бюджет {budget}, факт {actual:.2f}"


def test_confirmation_does_not_cross_segment_boundary():
    """Серия «три подряд» не склеивает отрезки, между которыми вырезаны окна."""
    from stad.metrics.event_level import confirm_alarms

    fired = np.ones((6, 1), dtype=bool)
    glued = confirm_alarms(fired, 3).ravel().tolist()
    split = confirm_alarms(fired, 3, segments=np.array([0, 0, 1, 1, 1, 1])).ravel().tolist()
    assert glued == [False, False, True, True, True, True]
    assert split == [False, False, False, False, True, True]


def test_contiguous_segments_break_on_gaps():
    from stad.metrics import contiguous_segments

    t = np.array([0, 1, 2, 10, 11, 12, 40.0])
    assert contiguous_segments(t, 1.0).tolist() == [0, 0, 0, 1, 1, 1, 2]


def test_calibration_requires_validation_scores():
    """Без score валидации порог не калибруется вовсе — тест подставлять нельзя."""
    import dataclasses

    from stad.data import make_synthetic_corridor
    from stad.metrics import full_report, make_calibration

    d = make_synthetic_corridor(stations=5, lanes=3, days=2, step_min=1.0,
                                n_events=6, window=8, seed=1)
    scores = np.zeros(d.y_test.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="калибруется на валидации"):
        full_report(scores, d, val_scores=None)
    with pytest.raises(ValueError, match="t_val"):
        make_calibration(np.zeros((len(d.X_val), d.n_nodes)),
                         dataclasses.replace(d, t_val=None))


@pytest.mark.parametrize("reduce", ["max", "q99", "q95", "mean"])
def test_node_reduction_shapes_and_order(reduce):
    """Агрегация по узлам сохраняет ось окон; max >= q99 >= q95 по построению."""
    from types import SimpleNamespace

    from stad.metrics import evaluation_view

    rng = np.random.default_rng(0)
    s = rng.standard_normal((50, 40))
    data = SimpleNamespace(
        meta={"labels_are_corridor_level": True},
        y_test=np.zeros((50, 40), dtype=int), event_id_test=-np.ones((50, 40), dtype=int),
    )
    out, y, eid = evaluation_view(s, data, reduce=reduce)
    assert out.shape == (50, 1) and y.shape == (50, 1) and eid.shape == (50, 1)

    mx = evaluation_view(s, data, reduce="max")[0]
    q99 = evaluation_view(s, data, reduce="q99")[0]
    q95 = evaluation_view(s, data, reduce="q95")[0]
    assert (mx >= q99 - 1e-12).all() and (q99 >= q95 - 1e-12).all()

    with pytest.raises(ValueError, match="неизвестная агрегация"):
        evaluation_view(s, data, reduce="median")
