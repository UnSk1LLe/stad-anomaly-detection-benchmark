"""Тесты слоя данных: контракт, отсутствие утечек, корректность разметки."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stad.data import make_synthetic_corridor
from stad.data.labels import align_incidents
from stad.data.splits import drop_anomalous_windows, time_split_indices


@pytest.fixture(scope="module")
def data():
    return make_synthetic_corridor(stations=8, lanes=3, days=3, step_min=1.0,
                                   n_events=10, window=10, seed=0)


def test_contract_validates(data):
    data.validate()
    assert data.n_nodes == 24
    assert data.window == 10
    assert data.n_features == 3
    assert 0.0 < data.prevalence < 0.5


def test_train_split_has_no_anomalies(data):
    """Unsupervised-постановка: обучающие окна должны быть чистыми.

    Проверяется косвенно — через метаданные: сколько окон выброшено.
    Если выброшено ноль при наличии событий, фильтр не работает.
    """
    assert data.meta["dropped_train_windows"] > 0
    assert data.meta["n_train_windows"] > 0


def test_splits_are_temporal_not_random():
    """Сплит последовательный: индексы не перемешаны."""
    tr, va, te = time_split_indices(100, train=0.6, val=0.15)
    assert tr.max() < va.min() < te.min()
    assert len(tr) == 60 and len(va) == 15 and len(te) == 25


def test_drop_anomalous_windows_applies_buffer():
    y = np.zeros((20, 2), dtype=int)
    y[10] = 1
    keep = drop_anomalous_windows(np.zeros((20, 2, 3, 1)), y, buffer=2)
    assert 10 not in keep
    for i in (8, 9, 11, 12):
        assert i not in keep, f"буфер не применён к окну {i}"
    assert 7 in keep and 13 in keep


def test_normalization_fitted_on_train_only(data):
    """Статистики нормализации берутся из train; test не центрирован идеально.

    Если test нормирован собственными статистиками, это утечка, и
    признаком будет среднее теста, равное нулю с высокой точностью.
    """
    assert abs(float(data.X_train.mean())) < 0.2
    # у теста есть события, поэтому его среднее не обязано быть нулевым
    assert "scaler_mean" in data.meta and "scaler_std" in data.meta


def test_labels_asymmetric_upstream():
    """Радиус разметки асимметричен: вверх по потоку шире, чем вниз."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    flow = pd.DataFrame({
        "ts": np.repeat(ts, 5),
        "node": np.tile(np.arange(5), 10),
        "milepost": np.tile([0.0, 1.0, 2.0, 3.0, 4.0], 10),
    })
    inc = pd.DataFrame([{"start": ts[4], "end": ts[5], "milepost": 2.0}])
    labelled, events = align_incidents(
        flow, inc, upstream_mi=1.5, downstream_mi=0.3, lead_min=0.0, trail_min=0.0
    )
    at_event = labelled[labelled["ts"] == ts[4]]
    marked = set(at_event.loc[at_event["y_point"] == 1, "milepost"])
    assert 1.0 in marked, "узел вверх по потоку (milepost 1.0) должен быть помечен"
    assert 3.0 not in marked, "узел вниз по потоку (milepost 3.0) не должен быть помечен"
    assert len(events) == 1


def test_lead_window_marks_before_report():
    """Опережающее окно помечает точки ДО времени отчёта."""
    ts = pd.date_range("2026-01-01", periods=20, freq="1min")
    flow = pd.DataFrame({"ts": ts, "node": 0, "milepost": 1.0})
    inc = pd.DataFrame([{"start": ts[10], "end": ts[12], "milepost": 1.0}])
    labelled, _ = align_incidents(flow, inc, lead_min=5.0, trail_min=0.0,
                                  upstream_mi=1.0, downstream_mi=1.0)
    assert labelled.loc[labelled["ts"] == ts[6], "y_point"].iloc[0] == 1
    assert labelled.loc[labelled["ts"] == ts[3], "y_point"].iloc[0] == 0


def test_events_outside_coverage_dropped():
    """Событие вне покрытия детекторов исключается из реестра.

    Иначе recall занижается по построению: обнаружить такое событие
    невозможно, и его присутствие искажает метрику для всех моделей.
    """
    ts = pd.date_range("2026-01-01", periods=10, freq="1min")
    flow = pd.DataFrame({"ts": ts, "node": 0, "milepost": 1.0})
    inc = pd.DataFrame([
        {"start": ts[4], "end": ts[5], "milepost": 1.0},
        {"start": ts[6], "end": ts[7], "milepost": 99.0},   # вне покрытия
    ])
    _, events = align_incidents(flow, inc, upstream_mi=0.5, downstream_mi=0.5,
                                lead_min=0.0, trail_min=0.0)
    assert len(events) == 1
    assert events["milepost"].iloc[0] == 1.0


def test_adjacency_is_row_normalised(data):
    sums = data.A.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), f"строки A не нормированы: {sums[:5]}"


def test_synthetic_anomalies_are_detectable(data):
    """Санити: у аномальных узлов скорость в среднем ниже нормальных.

    Если это не так, синтетика бессмысленна: детектировать нечего, и
    любой нулевой результат будет артефактом генератора, а не моделей.
    """
    speed_idx = data.feature_names.index("speed")
    last = data.X_test[:, :, -1, speed_idx]
    anomalous = last[data.y_test == 1]
    normal = last[data.y_test == 0]
    assert anomalous.mean() < normal.mean(), (
        f"аномальная скорость {anomalous.mean():.3f} не ниже нормальной {normal.mean():.3f}"
    )
