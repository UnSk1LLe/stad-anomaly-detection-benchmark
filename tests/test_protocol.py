"""Тесты протокола на уровне пайплайна: калибровка порога и предзарегистрированное подмножество."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stad.baselines.controls import RandomScorer
from stad.data import make_synthetic_corridor
from stad.metrics.stats import nemenyi_cd
from stad.registry import CORE, PRIMARY_COMPARISON, REFERENCE, Config
from stad.report import primary_runs, rank_table, validate_protocol
from stad.runner import run_config
from stad.train import TrainConfig


@pytest.fixture(scope="module")
def data():
    return make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)


def test_calibration_windows_are_timestamped(data):
    """Калибровочная выборка привязана ко времени и лежит строго до теста (rolling origin)."""
    assert data.t_calib is not None and len(data.t_calib) == len(data.X_calib)
    assert data.t_calib.max() < data.t_test.min()
    # буферы вокруг событий не вырезаются, поэтому выборка не меньше чистой валидации
    assert len(data.X_calib) >= len(data.X_val)


def test_random_scorer_calibration_stream_independent(data):
    """Score случайного контроля на валидации не должен быть началом тестового ряда."""
    sc = RandomScorer(seed=0)
    test, calib = sc.score(data), sc.score_calib(data)
    assert calib.shape == (len(data.X_calib), data.n_nodes)
    n = min(len(test), len(calib))
    assert not np.allclose(test[:n], calib[:n])


def test_pipeline_threshold_ignores_test_labels(data):
    """Через весь run_config: перемешать метки теста — порог не меняется."""
    cfg = Config(name="b_snd", group="baseline", baseline="snd", rationale="тест")
    kw = dict(seed=0, dataset_name="d", train_cfg=TrainConfig(), param_budget=1000,
              alarm_budget_per_hour=1.0)
    row, _, _, _, calib_scores = run_config(cfg, data, **kw)

    perm = np.random.default_rng(0).permutation(len(data.y_test))
    shuffled = dataclasses.replace(data, y_test=data.y_test[perm],
                                   event_id_test=data.event_id_test[perm])
    row2, *_ = run_config(cfg, shuffled, **kw)

    assert row["threshold_source"] == "validation"
    assert row["threshold"] == row2["threshold"]
    assert calib_scores.shape == (len(data.X_calib), data.n_nodes)


def test_trained_detector_returns_calibration_scores(data):
    from stad.model import build_detector
    from stad.train import train_detector

    det = build_detector("gcn_gru", "recon", hidden=8, n_features=data.n_features,
                         n_nodes=data.n_nodes, window=data.window)
    out = train_detector(det, data, TrainConfig(epochs=1, batch_size=64, device="cpu"), seed=0)
    assert out.calib_scores is not None
    assert out.calib_scores.shape == (len(data.X_calib), data.n_nodes)
    assert np.isfinite(out.calib_scores).all()


# ------------------------------------------------- предзарегистрированное подмножество
def _fake_runs(n_folds: int = 4, n_seeds: int = 5, *, seed: int = 0) -> pd.DataFrame:
    """12 конфигураций ядра; у кандидатов padf растёт с индексом, у остальных — шум."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_folds):
        for s in range(n_seeds):
            for c in CORE:
                in_primary = c.name in PRIMARY_COMPARISON
                level = 0.15 + 0.1 * PRIMARY_COMPARISON.index(c.name) if in_primary else 0.05
                rows.append({
                    "config": c.name, "label": c.name, "group": c.group,
                    "group_label": c.group, "dataset": f"fold{f}", "seed": s,
                    "block": f"fold{f}|s{s}",
                    "padf": level + rng.normal(0, 0.01),
                    "average_precision": 0.3, "event_recall": 0.2,
                    "pa_f1_INVALID_for_ranking": 0.8,
                    "budget_within_tolerance": True,
                    "alarms_per_hour": 0.25, "ap_lift_over_random": 1.0, "fpr_observed": 0.01,
                    "inference_ms_per_window": 1.0, "median_delay_min": 0.0,
                    "n_params": 1000, "pa_inflation_ratio": 3.0,
                })
    return pd.DataFrame(rows)


def test_primary_comparison_is_registered_and_contains_reference():
    names = {c.name for c in CORE}
    assert REFERENCE in PRIMARY_COMPARISON
    assert set(PRIMARY_COMPARISON) <= names
    assert len(set(PRIMARY_COMPARISON)) == len(PRIMARY_COMPARISON)
    # контроли и бейзлайны в выводной тест не входят
    for c in CORE:
        if c.group in {"control", "baseline"}:
            assert c.name not in PRIMARY_COMPARISON


def test_nemenyi_cd_shrinks_with_fewer_methods():
    """Арифметика, ради которой подмножество и вводится: 12 методов → 3.73, 6 → 1.69 при 20 блоках."""
    assert nemenyi_cd(12, 20) == pytest.approx(3.73, abs=0.01)
    assert nemenyi_cd(6, 20) == pytest.approx(1.69, abs=0.01)


def test_power_check_uses_primary_subset_only():
    runs = _fake_runs()
    sub, note = primary_runs(runs)
    assert set(sub["config"]) == set(PRIMARY_COMPARISON) and note == ""

    checks = {c.name: c for c in validate_protocol(runs, prevalence=0.25)}
    power = checks["Мощность: критическая разница меньше размаха рангов"]
    assert "6 методах" in power.detail and "20 блоках" in power.detail
    assert power.passed


def test_rank_table_ranks_only_primary_descriptive_for_rest():
    table = rank_table(_fake_runs())
    ranked = table[table["mean_rank"].notna()]
    assert set(ranked["config"]) == set(PRIMARY_COMPARISON)
    # бейзлайны и контроли остаются в таблице — описательно, без ранга
    assert {"ctrl_random", "base_pca", "base_california"} <= set(table["config"])
    assert table[~table["config"].isin(PRIMARY_COMPARISON)]["mean_rank"].isna().all()
    # ранги внутри подмножества: от 1 до 6
    assert ranked["mean_rank"].min() >= 1 and ranked["mean_rank"].max() <= 6


def test_primary_runs_falls_back_loudly_when_subset_absent():
    runs = _fake_runs()
    smoke_like = runs[runs["config"].isin(["ctrl_random", "base_pca", REFERENCE])]
    sub, note = primary_runs(smoke_like)
    assert len(sub) == len(smoke_like) and "ВСЕМ конфигурациям" in note


def test_random_control_check_is_not_weakened():
    """Контроль выше половины обученных по-прежнему валит критическую проверку."""
    runs = _fake_runs()
    runs.loc[runs["config"] == "ctrl_random", "padf"] = 0.9
    checks = {c.name: c for c in validate_protocol(runs, prevalence=0.25)}
    assert not checks["Случайный контроль не обходит обученные модели"].passed
