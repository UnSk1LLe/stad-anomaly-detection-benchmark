"""Диагностика прогона: единицы тревог, нулевое распределение, bootstrap по событиям."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stad.data import make_synthetic_corridor
from stad.diagnostics import (
    TZ_REFERENCE,
    Protocol,
    _Cells,
    _mc_bound_text,
    _tz_computed,
    alarm_episode_starts,
    alarm_units,
    alarm_units_calib,
    alarm_units_test,
    block_ci_mc_range,
    cluster_bootstrap,
    dataset_mismatches,
    event_credits,
    event_level_comparison,
    fold_level_comparison,
    model_free_scores,
    oracle_report,
    random_band,
    random_null,
    run_diagnostics,
    seed_variance,
    stratified_resample,
    tz_reconciliation,
)
from stad.eventlog import event_table
from stad.experiment import ExperimentConfig
from stad.metrics import (
    alarms_per_hour,
    contiguous_segments,
    eval_segments,
    evaluation_view,
    full_report,
    threshold_at_alarm_rate,
)
from stad.registry import REFERENCE, Config, get_grid
from stad.report import compare_to_reference
from stad.runner import run_config, run_grid
from stad.train import TrainConfig

PROTO = Protocol(alarm_budget_per_hour=1.0, half_life_min=15.0, persistence=3, node_reduce="max")
CSVS = ("alarm_units", "alarm_units_summary", "random_null", "random_band", "model_free", "seed_variance",
        "pairwise", "reproduction", "oracle_rates")


@pytest.fixture(scope="module")
def data():
    # 5 событий в тесте: кластерный bootstrap по событиям не вырожден
    return make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=20, window=8, seed=2)


def _units(s: np.ndarray, y: np.ndarray | None = None, segments: np.ndarray | None = None) -> dict:
    s = np.asarray(s, dtype=float)[:, None]
    y = np.zeros(s.shape, dtype=int) if y is None else np.asarray(y)[:, None]
    return alarm_units(s, y, 0.5, step_min=1.0, persistence=3, segments=segments)


# ------------------------------------------------------------ эпизоды
def test_ten_consecutive_alarm_windows_are_one_episode():
    assert alarm_episode_starts(np.r_[np.zeros(3), np.ones(10), np.zeros(3)]).sum() == 1
    # 12 превышений подряд при подтверждении 3 окнами: 10 окон тревоги, один эпизод
    u = _units(np.r_[np.zeros(3), np.ones(12), np.zeros(3)])
    assert (u["alarm_windows"], u["episodes"]) == (10, 1)


def test_runs_separated_by_quiet_window_are_two_episodes():
    a = np.r_[np.ones(4), 0, np.ones(4)]
    assert alarm_episode_starts(a).sum() == 2


def test_run_crossing_segment_boundary_is_two_episodes():
    a = np.ones(10)
    seg = np.r_[np.zeros(5), np.ones(5)].astype(int)
    assert alarm_episode_starts(a, seg).sum() == 2
    assert alarm_episode_starts(a).sum() == 1


def test_alarm_windows_inside_events_are_not_counted():
    s = np.ones(20)
    y = np.zeros(20, dtype=int)
    y[8:12] = 1
    u = _units(s, y)
    # подтверждены окна 2..19; из них 8..11 — внутри события: остаются 2..7 и 12..19
    assert u["alarm_windows"] == 14
    assert u["episodes"] == 2


def test_alarm_cells_per_hour_equals_metric(data):
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(data.y_test.shape).astype(np.float32)
    calib = rng.standard_normal((len(data.X_calib), data.n_nodes)).astype(np.float32)
    rep = full_report(scores, data, calib_scores=calib, **PROTO.report_kwargs())
    u = alarm_units_test(scores, data, rep["threshold"], persistence=3, node_reduce="max")
    s, y, _ = evaluation_view(scores, data, reduce="max")
    direct = alarms_per_hour(s, y, rep["threshold"], float(data.meta["step_min"]), persistence=3,
                             segments=eval_segments(data))
    assert u["alarm_cells_per_hour"] == pytest.approx(direct, abs=1e-12)
    assert u["alarm_cells_per_hour"] == pytest.approx(rep["alarms_per_hour"], abs=1e-12)
    # на поузловых метках окно тревоги объединяет ячейки: окон не больше, чем ячеек
    assert u["alarm_windows"] <= u["alarm_cells"]
    assert u["episodes"] <= u["alarm_windows"]


def test_calib_units_confirm_within_segment_only(data):
    seg = contiguous_segments(data.t_calib)
    b = int(np.flatnonzero(np.diff(seg))[0]) + 1            # первое окно второго сегмента
    calib = np.zeros((len(data.X_calib), data.n_nodes), dtype=np.float32)
    calib[b - 2:b + 2] = 1.0                                # 4 окна подряд, но через разрыв: 2 + 2
    calib[b + 20] = 1.0                                     # одиночное превышение
    u = alarm_units_calib(calib, data, 0.5, persistence=3, node_reduce="max")
    assert (u["alarm_windows"], u["episodes"]) == (0, 0)
    calib[b + 2] = 1.0                                      # во втором сегменте теперь 3 окна подряд
    u = alarm_units_calib(calib, data, 0.5, persistence=3, node_reduce="max")
    assert (u["alarm_windows"], u["episodes"]) == (1, 1)
    assert u["hours"] == pytest.approx(len(data.X_calib) * float(data.meta["step_min"]) / 60.0)


def test_oracle_rate_is_met_on_test_normals(data):
    rng = np.random.default_rng(4)
    scores = rng.standard_normal(data.y_test.shape).astype(np.float32)
    o = oracle_report(scores, data, 1.0, PROTO)
    s, y, _ = evaluation_view(scores, data, reduce="max")
    step = float(data.meta["step_min"])
    thr = threshold_at_alarm_rate(s, y, 1.0, step, persistence=3, segments=eval_segments(data))
    assert o["threshold_ORACLE"] == pytest.approx(thr)
    got = alarms_per_hour(s, y, thr, step, persistence=3, segments=eval_segments(data))
    assert o["alarms_per_hour_ORACLE"] == pytest.approx(got, abs=1e-12)
    assert 0 < o["alarms_per_hour_ORACLE"] <= 1.0


# ---------------------------------------------------- случайный контроль
def test_random_null_reproduces_grid_cells(data):
    null = random_null({"d": data}, range(2), PROTO, oracle_rates=(1.0,))
    cfg = Config(name="ctrl_random", group="control", baseline="random", rationale="тест")
    for seed in (0, 1):
        row, *_ = run_config(cfg, data, seed=seed, dataset_name="d", train_cfg=TrainConfig(),
                             param_budget=1000, alarm_budget_per_hour=1.0)
        got = null[null["seed"] == seed].iloc[0]
        assert got["padf"] == row["padf"]
        assert got["threshold"] == row["threshold"]
        assert got["alarms_per_hour"] == row["alarms_per_hour"]
    assert "padf_ORACLE_1ph" in null.columns


def test_random_band_one_draw_wider_than_many():
    rng = np.random.default_rng(0)
    null = pd.DataFrame({"dataset": np.repeat(["a", "b", "c", "d"], 200),
                         "padf": rng.uniform(0, 0.3, 800)})
    b5, b1 = random_band(null, 5, n_mc=4000), random_band(null, 1, n_mc=4000)
    assert b5["expected"] == pytest.approx(null["padf"].mean())
    assert b1["q_lo"] < b5["q_lo"] < b5["expected"] < b5["q_hi"] < b1["q_hi"]


def test_random_band_draws_within_each_fold():
    # фолды с непересекающимися носителями: при одном значении на фолд среднее всегда 0.5;
    # выборка из общего пула дала бы и 0, и 1
    null = pd.DataFrame({"dataset": np.repeat(["a", "b"], 50), "padf": np.r_[np.zeros(50), np.ones(50)]})
    b1 = random_band(null, 1, n_mc=2000)
    assert b1["q_lo"] == b1["q_hi"] == 0.5


def test_seed_variance_undefined_with_one_seed():
    runs = pd.DataFrame({"config": ["a", "a", "b", "b"], "dataset": ["f0", "f1", "f0", "f1"],
                         "padf": [0.1, 0.2, 0.3, 0.4]})
    sv = seed_variance(runs)
    assert sv["zero_within_fold_variance"].isna().all()


def test_seed_variance_flags_deterministic_config():
    runs = pd.DataFrame({
        "config": ["det"] * 4 + ["rnd"] * 4,
        "dataset": ["f0", "f0", "f1", "f1"] * 2,
        "padf": [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.1, 0.0],
    })
    sv = seed_variance(runs).set_index("config")
    assert bool(sv.loc["det", "zero_within_fold_variance"])
    assert not bool(sv.loc["rnd", "zero_within_fold_variance"])
    assert sv.loc["det", "independent_values"] == 2 and sv.loc["rnd", "independent_values"] == 4


# ------------------------------------------------- bootstrap по событиям
def test_stratified_resample_keeps_fold_event_counts():
    strata = np.array(["f0"] * 3 + ["f1"] * 7 + ["f2"])
    idx = stratified_resample(strata, 50, np.random.default_rng(1))
    assert idx.shape == (50, len(strata))
    want = pd.Series(strata).value_counts().sort_index()
    for row in idx:
        got = pd.Series(strata[row]).value_counts().sort_index()
        pd.testing.assert_series_equal(got, want)


def test_cluster_bootstrap_two_sided_p():
    strata = np.repeat(["a", "b"], 6)
    neg = cluster_bootstrap(np.full(12, -0.1), strata, n_boot=300)
    assert neg["mean_diff"] == pytest.approx(-0.1) and neg["p_value"] == 0.0
    d = np.r_[0.3, -0.1, 0.2, -0.2, 0.1, 0.0, -0.3, 0.4, 0.1, -0.1, 0.2, 0.0]
    res = cluster_bootstrap(d, strata, n_boot=400, seed=5)
    boots = d[stratified_resample(strata, 400, np.random.default_rng(5))].mean(axis=1)
    want = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    assert res["p_value"] == pytest.approx(want)
    assert res["ci_lo"] == pytest.approx(np.quantile(boots, 0.025))


def test_fold_comparison_averages_seeds_within_fold():
    runs = pd.DataFrame({
        "config": ["a", "a", "a", REFERENCE, REFERENCE, REFERENCE],
        "dataset": ["f0", "f0", "f1", "f0", "f0", "f1"],
        "block": ["f0/0", "f0/1", "f1/0", "f0/0", "f0/1", "f1/0"],
        "padf": [0.3, 0.1, 0.5, 0.0, 0.0, 0.0],
    })
    res = fold_level_comparison(runs, n_boot=200).set_index("config")
    # фолды: 0.2 и 0.5 -> 0.35 (среднее по блокам дало бы 0.3)
    assert res.loc["a", "fold_mean_diff"] == pytest.approx(0.35)
    assert res.loc["a", "fold_n"] == 2 and res.loc["a", "fold_n_positive"] == 2


def test_event_comparison_requires_same_seeds():
    ev = pd.DataFrame({"dataset": ["f0"] * 3, "event_id": [0, 1, 2]})
    rows = [ev.assign(config=REFERENCE, seed=k, credit=0.5) for k in (0, 1)]
    rows.append(ev.assign(config="one_seed", seed=0, credit=0.5))
    rows += [ev.assign(config="both", seed=k, credit=0.5) for k in (0, 1)]
    cmp = event_level_comparison(event_credits(pd.concat(rows)), n_boot=100, n_seeds=2).set_index("config")
    assert not bool(cmp.loc["one_seed", "event_complete"]) and bool(cmp.loc["both", "event_complete"])
    assert cmp.loc["one_seed", "event_n_seeds_min"] == 1


def test_dataset_mismatch_against_manifest(data):
    same = {"datasets": {"syn": {k: data.meta[k] for k in ("step_min", "n_test_windows", "n_events_test")}}}
    assert dataset_mismatches({"syn": data}, same) == []
    other = json.loads(json.dumps(same))
    other["datasets"]["syn"]["n_events_test"] = 99
    bad = dataset_mismatches({"syn": data}, other)
    assert len(bad) == 1 and "n_events_test" in bad[0]


def test_cluster_bootstrap_identical_credits_give_zero():
    res = cluster_bootstrap(np.zeros(12), np.repeat(["a", "b", "c"], 4), n_boot=500)
    assert res["mean_diff"] == 0 and res["ci_lo"] == 0 and res["ci_hi"] == 0
    assert res["p_value"] == 1.0 and res["half_width"] == 0


def test_event_comparison_constant_shift():
    rng = np.random.default_rng(3)
    ev = pd.DataFrame({"dataset": np.repeat(["f0", "f1"], [5, 6]), "event_id": np.arange(11)})
    base = rng.uniform(0, 0.8, len(ev))
    rows = []
    for seed in (0, 1):
        rows.append(ev.assign(config=REFERENCE, seed=seed, credit=base))
        rows.append(ev.assign(config="same", seed=seed, credit=base))
        rows.append(ev.assign(config="shift", seed=seed, credit=base + 0.1))
    credits = event_credits(pd.concat(rows, ignore_index=True))
    assert (credits["n_seeds"] == 2).all()
    cmp = event_level_comparison(credits, n_boot=500).set_index("config")
    assert cmp.loc["same", "event_mean_diff"] == 0
    assert cmp.loc["same", "event_ci_lo"] == 0 and cmp.loc["same", "event_ci_hi"] == 0
    d = cmp.loc["shift"]
    assert d["event_mean_diff"] == pytest.approx(0.1)
    assert d["event_ci_lo"] - 1e-12 <= 0.1 <= d["event_ci_hi"] + 1e-12
    assert d["event_n_events"] == 11 and bool(d["event_complete"])


# -------------------------------------------------- безмодельные score
def test_model_free_scores_broadcast():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7, 4, 5, 3)).astype(np.float32)
    names = ["speed", "occupancy", "volume"]
    occ = model_free_scores(X, names, "occ_mean")
    assert occ.shape == (7, 4)
    assert np.allclose(occ, occ[:, :1])                  # одно значение на окно, на всех узлах
    assert np.allclose(occ.max(axis=1), X[:, :, -1, 1].mean(axis=1))
    spd = model_free_scores(X, names, "neg_speed_mean")
    assert np.allclose(spd[:, 0], -X[:, :, -1, 0].mean(axis=1))
    with pytest.raises(ValueError, match="occupancy"):
        model_free_scores(X, ["speed", "flow", "volume"], "occ_mean")


# ------------------------------------------------------------ сверка с ТЗ
def _tz_ctx() -> dict:
    """Минимальный контекст отчёта с известными агрегатами (два фолда, два сида)."""
    runs = pd.DataFrame({
        "config": ["a"] * 4 + ["ctrl_random"] * 4,
        "dataset": ["f0", "f0", "f1", "f1"] * 2,
        "seed": [0, 1] * 4,
        "padf": [0.2, 0.2, 0.4, 0.4, 0.1, 0.0, 0.2, 0.1],
        "average_precision": 0.3, "roc_auc_reference_only": 0.6,
        "alarms_per_hour": [1.0, 3.0, 0.0, 0.0, 0.5, 0.5, 0.25, 0.75],
    })
    meta = {"labels_are_corridor_level": True, "step_min": 1.0}
    datasets = {"f0": SimpleNamespace(meta=meta, t_calib=np.arange(600), events=[0, 1, 2]),
                "f1": SimpleNamespace(meta=meta, t_calib=np.arange(720), events=[0, 1, 2, 3])}
    units = runs[["config", "dataset", "seed"]].assign(
        episodes_per_hour=[0.1, 0.3, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2], calib_episodes=[1, 2, 1, 1, 2, 2, 1, 1])
    cells = _Cells(units=units, events=pd.DataFrame(), reproduction=pd.DataFrame(),
                   oracle=pd.DataFrame(columns=["config", "dataset", "seed", "rate_per_hour"]),
                   missing=pd.DataFrame(columns=["config", "dataset", "seed", "reason"]))
    rnd = runs[runs["config"] == "ctrl_random"]
    null = pd.concat([
        rnd[["dataset", "seed", "padf"]].assign(padf_runs=rnd["padf"], in_run=True,
                                                padf_ORACLE_1ph=[0.3, 0.5, 0.1, 0.3]),
        pd.DataFrame({"dataset": ["f0", "f1"], "seed": [2, 2], "padf": [0.6, 0.6], "padf_runs": np.nan,
                      "in_run": False, "padf_ORACLE_1ph": [0.9, 0.9]}),
    ], ignore_index=True)
    return {
        "runs": runs, "cells": cells, "null": null, "datasets": datasets,
        "protocol": Protocol(alarm_budget_per_hour=0.25, half_life_min=15.0, persistence=3, node_reduce="max"),
        "model_free": pd.DataFrame({"score": ["occ_mean", "occ_mean"], "average_precision": [0.3, 0.4],
                                    "roc_auc_reference_only": [0.7, 0.8]}),
        "pairwise": pd.DataFrame({"config": ["a", "ctrl_random"], "event_complete": [True, True],
                                  "event_half_width": [0.1, 0.15]}),
        "band_k": {"n_draws_per_fold": 2, "q_lo": 0.05, "q_hi": 0.2},
        "band_1": {"n_draws_per_fold": 1, "q_lo": 0.01, "q_hi": 0.3},
        "seedvar": seed_variance(runs),
    }


def test_tz_computed_aggregations():
    ctx = _tz_ctx()
    c, band_rows, zero = _tz_computed(ctx)
    assert c["n_events_test"] == 7
    assert c["calib_hours_mean"] == pytest.approx(11.0)                      # 10 ч и 12 ч
    assert (c["calib_budget_windows_min"], c["calib_budget_windows_max"]) == (2, 3)   # floor(0.25 · ч)
    # частота тревог: среднее по сидам в фолде (a/f0 = 2.0) и среднее конфигурации (a = 1.0)
    assert (c["alarm_windows_per_hour_fold_min"], c["alarm_windows_per_hour_fold_max"]) == (0.0, 2.0)
    assert (c["alarm_windows_per_hour_config_min"], c["alarm_windows_per_hour_config_max"]) == (0.5, 1.0)
    assert c["episodes_per_hour_config_min"] == pytest.approx(0.1)
    assert c["episodes_per_hour_config_max"] == pytest.approx(0.2)
    assert (c["calib_episodes_min"], c["calib_episodes_max"]) == (1, 2)
    assert c["random_expected_padf"] == pytest.approx(ctx["null"]["padf"].mean())
    # ORACLE ctrl_random без score клеток — только сиды прогона из нулевого распределения
    assert c["random_oracle_padf_1"] == pytest.approx(0.3)
    assert c["model_free.occ_mean.average_precision"] == pytest.approx(0.35)
    assert (c["event_half_width_min"], c["event_half_width_max"]) == (0.1, 0.15)
    assert band_rows == {"2 на фолд": (0.05, 0.2), "1 на фолд": (0.01, 0.3)}
    assert sorted(zero) == ["a"]                                # у a сиды фолда дают одно число

    ctx["pairwise"].loc[1, "event_complete"] = False            # одна пара неполная -> MDE нет
    ctx["null"].loc[0, "padf"] += 0.01                          # нулевое не воспроизводит runs.csv
    ctx["cells"].missing.loc[0] = ["a", "f0", 0, "обучаемая: нужен scores/ прогона"]
    c2, _, _ = _tz_computed(ctx)
    for key in ("event_half_width_min", "random_oracle_padf_1", "episodes_per_hour_config_min",
                "calib_episodes_max"):
        assert key not in c2, key
    assert c2["alarm_windows_per_hour_fold_max"] == 2.0         # по runs.csv считается и так


def test_tz_reconciliation_precision_and_status():
    c = {"n_events_test": 35.0, "event_half_width_min": 0.1049, "block_ci_lo.base_iforest": 0.0711,
         "random_expected_padf": 0.1043}
    t = tz_reconciliation(c, {"5 на фолд": (0.06, 0.15), "1 на фолд": (0.03, 0.2)}, None,
                          computed_notes={"random_expected_padf": "сидов: 50, фолдов: 1"}).set_index("пункт")

    def row(key: str) -> pd.Series:
        return t.loc[TZ_REFERENCE[key][0]]

    assert row("n_events_test")["ТЗ"] == "35" and row("n_events_test")["статус"].startswith("совпадает")
    assert row("calib_hours_mean")["ТЗ"] == "11"
    assert row("event_half_width_min")["ТЗ"] == "0.10"                 # точность ТЗ, а не :g
    assert row("event_half_width_min")["статус"] == "совпадает с точностью ТЗ (±0.005)"
    assert row("block_ci_lo.base_iforest")["ТЗ"] == "0.073"
    assert row("block_ci_lo.base_iforest")["статус"] == "расходится"
    assert row("random_expected_padf")["пересчёт"] == "0.1043 (сидов: 50, фолдов: 1)"
    assert row("random_expected_padf")["статус"] == "совпадает до 3-го знака"
    assert row("padf_mean.base_iforest")["статус"] == "нет score"
    band = t.loc["полоса случайного контроля, нижняя граница"]
    assert band["ТЗ"] == "0.040" and band["статус"].startswith("расходится")
    assert "между вариантами" in band["статус"]
    assert t.loc["нулевой разброс по сидам внутри фолда", "статус"].startswith("не определено")


def test_block_ci_mc_range_brackets_report_ci():
    rng = np.random.default_rng(1)
    blocks = [f"f{i}/{s}" for i in range(4) for s in range(5)]
    runs = pd.concat([
        pd.DataFrame({"config": "base_iforest", "label": "x", "block": blocks,
                      "padf": 0.3 + 0.1 * rng.standard_normal(20)}),
        pd.DataFrame({"config": REFERENCE, "label": "r", "block": blocks,
                      "padf": 0.2 + 0.1 * rng.standard_normal(20)}),
    ], ignore_index=True)
    mc = block_ci_mc_range(runs, "base_iforest", seeds=range(3), n_boots=(2000, 5000))
    ref = compare_to_reference(runs).set_index("config").loc["base_iforest"]   # сид 0, 5000 повторов
    assert mc["lo_min"] <= ref["ci_lo"] <= mc["lo_max"] and mc["hi_min"] <= ref["ci_hi"] <= mc["hi_max"]
    assert block_ci_mc_range(runs, "нет такой") == {}
    assert "не воспроизводится" in _mc_bound_text("Нижняя", 0.073, 0.0005, 0.0706, 0.0722)
    assert "шумом Monte Carlo" in _mc_bound_text("Нижняя", 0.073, 0.0005, 0.0706, 0.0726)


# ------------------------------------------------------------- сквозной
def test_run_diagnostics_end_to_end(data, tmp_path):
    grid = {c.name: c for c in get_grid("core")}
    configs = tuple(grid[n] for n in ("ctrl_random", "base_snd", REFERENCE))
    run = tmp_path / "run"
    run_grid(configs, {"syn": data}, seeds=(0, 1), param_budget=20_000, out_dir=run,
             train_cfg=TrainConfig(epochs=1, batch_size=64, device="cpu", patience=3),
             alarm_budget_per_hour=1.0, save_checkpoints=False, verbose=False)
    # одна клетка без score: небучаемая, должна пересчитаться
    for suffix in (".npy", "__calib.npy"):
        (run / "scores" / f"ctrl_random__syn__seed1{suffix}").unlink()

    cfg = ExperimentConfig(name="t", grid="core", seeds=(0, 1), param_budget=20_000,
                           alarm_budget_per_hour=1.0, out_dir=str(run))
    out = tmp_path / "diag"
    path = run_diagnostics(cfg, {"syn": data}, run, out, random_seeds=3, n_boot=200, n_mc=500,
                           oracle_rates=(1.0,), recompute_nontrainable=True, tz_reference=True)
    assert path.exists()
    for name in CSVS:
        assert (out / f"{name}.csv").exists(), name
    assert (out / "scores_recomputed" / "ctrl_random__syn__seed1.npy").exists()

    units = pd.read_csv(out / "alarm_units.csv")
    assert len(units) == 6
    np.testing.assert_allclose(units["alarm_cells_per_hour"], units["alarms_per_hour_runs"], atol=1e-12)
    np.testing.assert_allclose(units["threshold"], units["threshold_runs"], rtol=1e-6)

    rep = pd.read_csv(out / "reproduction.csv")
    assert rep["padf_match"].all()
    src = rep.set_index(["config", "seed"])["source"]
    assert src[("ctrl_random", 1)] == "recomputed" and src[("ctrl_random", 0)] == "saved"

    null = pd.read_csv(out / "random_null.csv")
    chk = null[null["in_run"]]
    assert len(chk) == 2 and (chk["padf"] == chk["padf_runs"]).all()

    pw = pd.read_csv(out / "pairwise.csv").set_index("config")
    assert set(pw.index) == {"ctrl_random", "base_snd"}
    assert pw["event_mean_diff"].notna().all() and pw["block_mean_diff"].notna().all()
    assert pw["event_complete"].all() and (pw["event_half_width"] > 0).all()
    assert "block_verdict" not in pw.columns

    md = path.read_text(encoding="utf-8")
    for head in ("## A.", "## B.", "## C. ORACLE", "## D.", "## E.", "## F.", "ТЗ v2 §1 -> пересчёт",
                 "## Чего не хватает"):
        assert head in md
    assert "Score есть у всех клеток" in md

    # формат TZ 1.2: events.csv прогона есть, scores/ референса нет — кредиты берутся из events.csv,
    # сравнение по событиям и минимально различимый эффект остаются посчитанными
    runs = pd.read_csv(run / "runs.csv")
    tables = []
    for r in runs.itertuples():
        name = f"{r.config}__{r.dataset}__seed{r.seed}.npy"
        f = run / "scores" / name
        s = np.load(f if f.exists() else out / "scores_recomputed" / name)
        t = event_table(s, data, float(r.threshold), half_life_min=15.0, persistence=3)
        tables.append(t.assign(config=r.config, dataset=r.dataset, seed=int(r.seed)))
    pd.concat(tables, ignore_index=True).to_csv(run / "events.csv", index=False)
    for seed in (0, 1):
        for suffix in (".npy", "__calib.npy"):
            (run / "scores" / f"{REFERENCE}__syn__seed{seed}{suffix}").unlink()
    out2 = tmp_path / "diag_events"
    md2 = run_diagnostics(cfg, {"syn": data}, run, out2, random_seeds=2, n_boot=200, n_mc=500,
                          oracle_rates=(1.0,), recompute_nontrainable=True,
                          tz_reference=True).read_text(encoding="utf-8")
    pw2 = pd.read_csv(out2 / "pairwise.csv").set_index("config")
    assert pw2["event_complete"].all()
    np.testing.assert_allclose(pw2["event_mean_diff"], pw.loc[pw2.index, "event_mean_diff"], atol=1e-12)
    assert "Кредиты событий взяты из `events.csv`" in md2
    hw_row = next(line for line in md2.splitlines() if "полуширина ДИ по событиям, минимум" in line)
    assert "нет score" not in hw_row
    assert "обучаемая: нужен scores/ прогона" in md2


def test_run_diagnostics_rejects_other_run(data, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"config": ["ctrl_random"], "dataset": ["syn"], "seed": [0], "padf": [0.1]}).to_csv(
        run / "runs.csv", index=False)
    meta = {"label_source": "both", "n_events_test": data.meta["n_events_test"]}
    (run / "manifest.json").write_text(json.dumps({"datasets": {"syn": meta}}), encoding="utf-8")
    cfg = ExperimentConfig(name="t", grid="core", seeds=(0,), out_dir=str(run))
    data_crash = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0, n_events=20,
                                         window=8, seed=2)
    data_crash.meta["label_source"] = "crash"
    with pytest.raises(ValueError, match="label_source"):
        run_diagnostics(cfg, {"syn": data_crash}, run, tmp_path / "diag", random_seeds=1)
