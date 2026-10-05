"""Пособытийная таблица складывается в padf клетки — те же функции, те же числа."""
from __future__ import annotations

import numpy as np
import pytest

from stad.data import make_synthetic_corridor
from stad.eventlog import EVENT_COLUMNS, event_table
from stad.registry import Config
from stad.runner import run_config
from stad.train import TrainConfig


@pytest.fixture(scope="module")
def data():
    return make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)


@pytest.mark.parametrize("baseline", ["random", "snd", "california"])
def test_credit_mean_equals_cell_padf(data, baseline):
    cfg = Config(name=f"b_{baseline}", group="baseline", baseline=baseline, rationale="тест")
    row, _, scores, _, _ = run_config(cfg, data, seed=0, dataset_name="d", train_cfg=TrainConfig(),
                                      param_budget=1000, alarm_budget_per_hour=1.0)
    ev = event_table(scores, data, row["threshold"], half_life_min=15.0, persistence=3)

    assert tuple(ev.columns) == EVENT_COLUMNS
    assert len(ev) == row["n_events"]
    assert int(ev["detected"].sum()) == row["n_detected"]
    assert ev["credit"].sum() / len(ev) == pytest.approx(row["padf"], abs=1e-12)
    # необнаруженное событие: задержки нет, кредит ноль
    miss = ev[~ev["detected"]]
    assert miss["delay_min"].isna().all() and (miss["credit"] == 0).all()
    # обнаруженное не позже отчёта получает полный кредит, позже — меньше единицы
    hit = ev[ev["detected"]]
    assert np.all(hit.loc[hit["delay_min"] <= 0, "credit"] == 1.0)
    assert np.all((hit.loc[hit["delay_min"] > 0, "credit"] < 1.0))
    assert ev["credit"].between(0, 1).all()


def test_corridor_level_view_is_used(data):
    """Для корридор-уровневых меток таблица строится по свёрнутому score, как и runs.csv."""
    import dataclasses

    n = data.n_nodes
    y = data.y_test.max(axis=1, keepdims=True)
    eid = data.event_id_test.max(axis=1, keepdims=True)
    corridor = dataclasses.replace(
        data, y_test=np.repeat(y, n, axis=1), event_id_test=np.repeat(eid, n, axis=1),
        meta={**data.meta, "labels_are_corridor_level": True},
    )
    corridor = dataclasses.replace(
        corridor, events=corridor.events[corridor.events["event_id"].isin(np.unique(eid[eid >= 0]))]
        .reset_index(drop=True),
    )
    cfg = Config(name="b_random", group="baseline", baseline="random", rationale="тест")
    row, _, scores, _, _ = run_config(cfg, corridor, seed=1, dataset_name="d", train_cfg=TrainConfig(),
                                      param_budget=1000, alarm_budget_per_hour=1.0)
    ev = event_table(scores, corridor, row["threshold"], half_life_min=15.0, persistence=3)
    assert ev["credit"].mean() == pytest.approx(row["padf"], abs=1e-12)
