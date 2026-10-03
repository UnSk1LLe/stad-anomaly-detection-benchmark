"""Тесты проверок чувствительности (этап 2) и абляции BiGAN (этап 5)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stad.ablation import COMPONENTS, ablate_bigan, bigan_components, summarise_ablation
from stad.checkpoints import checkpoint_path, save_detector
from stad.data import make_synthetic_corridor
from stad.metrics import full_report
from stad.model import build_detector
from stad.sensitivity import (
    BASE_WINDOW,
    TIE_BREAK,
    aggregation_table,
    choose_reducer,
    invariance_diagnostic,
    ranking_stability,
    relabel_test,
)
from stad.train import TrainConfig, train_detector


# ---------------------------------------------------------------- окно меток
def _corridor(n=400, n_nodes=6):
    """Корридор-уровневые метки: 3 события с t_report, окна по 1 минуте."""
    base = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)
    n = min(n, len(base.t_test))
    events = pd.DataFrame({"event_id": [0, 1, 2], "t_report": [60.0, 200.0, 330.0]})
    t = np.arange(n, dtype=float)
    return dataclasses.replace(
        base,
        X_test=base.X_test[:n], t_test=t, events=events,
        y_test=np.zeros((n, base.n_nodes), dtype=np.int64),
        event_id_test=-np.ones((n, base.n_nodes), dtype=np.int64),
        meta={**base.meta, "labels_are_corridor_level": True},
    )


def test_relabel_widens_positives_monotonically():
    d = _corridor()
    narrow = relabel_test(d, 10, 10)
    base = relabel_test(d, *BASE_WINDOW)
    wide = relabel_test(d, 20, 30)
    assert narrow.y_test.mean() < base.y_test.mean() < wide.y_test.mean()
    # метки корридор-уровневые: одинаковы по всем узлам
    assert (base.y_test == base.y_test[:, :1]).all()
    # граница окна: t_report=60, lead=15, trail=20 -> положительны t в [45, 80]
    pos = np.where(base.y_test[:, 0] == 1)[0]
    assert pos.min() == 45 and 80 in pos and 81 not in pos[pos < 100]


def test_relabel_rejects_node_level_labels():
    d = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0, n_events=8, window=8, seed=2)
    with pytest.raises(ValueError, match="корридор-уровневых"):
        relabel_test(d, 15, 20)


# ----------------------------------------------------------- выбор агрегации
def _runs(padf: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"config": list(padf), "padf": list(padf.values())})


def test_choose_reducer_picks_by_random_control_position():
    by = {
        "max": _runs({"ctrl_random": 0.15, "a": 0.10, "b": 0.12, "c": 0.30, "ctrl_untrained": 0.05}),
        "q99": _runs({"ctrl_random": 0.15, "a": 0.20, "b": 0.22, "c": 0.30, "ctrl_untrained": 0.05}),
    }
    choice = choose_reducer(by)
    assert choice.chosen == "q99"                       # random ниже всех обученных только здесь
    t = choice.table.set_index("reduce")
    assert t.loc["max", "n_beaten_by_random"] == 2 and t.loc["q99", "n_beaten_by_random"] == 0
    assert bool(t.loc["q99", "random_below_all_trained"])


def test_choose_reducer_ties_keep_incumbent_max():
    same = _runs({"ctrl_random": 0.15, "a": 0.2, "b": 0.3})
    choice = choose_reducer({"mean": same, "q95": same, "max": same, "q99": same})
    assert choice.chosen == "max" == TIE_BREAK[0]


def test_choose_reducer_reports_when_nothing_fixes_the_control():
    bad = _runs({"ctrl_random": 0.9, "a": 0.2, "b": 0.3})
    choice = choose_reducer({"max": bad, "q99": bad})
    assert "НИ ОДНА" in choice.reason and "не чинит" in choice.reason


def test_choice_ignores_padf_of_best_model():
    """Критерий — контроль, не лидер: рост лучшей модели выбор не меняет."""
    a = _runs({"ctrl_random": 0.15, "a": 0.10, "b": 0.20})
    b = _runs({"ctrl_random": 0.15, "a": 0.10, "b": 0.99})     # лучшая модель взлетела
    assert choose_reducer({"max": a, "q99": b}).chosen == "max"


def test_invariance_diagnostic_flags_trained_driven_selection():
    t = aggregation_table({
        "max": _runs({"ctrl_random": 0.15, "a": 0.10, "b": 0.12}),
        "q99": _runs({"ctrl_random": 0.15, "a": 0.30, "b": 0.35}),
    })
    assert "эквивалентен выбору по качеству" in invariance_diagnostic(t)


def test_ranking_stability_spearman_one_for_identical_rankings():
    mk = lambda scale: pd.DataFrame({
        "config": ["x", "y", "z"] * 2, "block": ["b0"] * 3 + ["b1"] * 3,
        "padf": [0.1, 0.2, 0.3] * 2 if scale == 1 else [0.2, 0.4, 0.6] * 2,
        "prevalence": 0.2, "n_events": 5,
    })
    s = ranking_stability({BASE_WINDOW: mk(1), (10.0, 10.0): mk(2)})
    assert (s["spearman_vs_base"] > 0.99).all()


# ------------------------------------------------------------- абляция BiGAN
@pytest.fixture(scope="module")
def trained_bigan(tmp_path_factory):
    data = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)
    det = build_detector("gcn_gru", "bigan", hidden=8, n_features=data.n_features,
                         n_nodes=data.n_nodes, window=data.window)
    out = train_detector(det, data, TrainConfig(epochs=1, batch_size=64, device="cpu"), seed=0)
    run = tmp_path_factory.mktemp("run")
    save_detector(det, checkpoint_path(run, "cand_gcngru_bigan", "synthetic", 0),
                  encoder="gcn_gru", head="bigan", hidden=8, n_features=data.n_features,
                  n_nodes=data.n_nodes, window=data.window, adjacency=data.A)
    (run / "scores").mkdir()
    np.save(run / "scores" / "cand_gcngru_bigan__synthetic__seed0.npy", out.scores)
    return data, det, run


def test_components_shapes_and_combined_matches_detector_score(trained_bigan):
    import torch

    data, det, _ = trained_bigan
    comps = bigan_components(det, data.X_test[:20])
    assert set(comps) == set(COMPONENTS)
    for v in comps.values():
        assert v.shape == (20, data.n_nodes)
    direct = det.score(torch.from_numpy(data.X_test[:20])).numpy()
    np.testing.assert_allclose(comps["combined"], direct, rtol=1e-4, atol=1e-5)


def test_ablate_bigan_runs_all_components_with_validation_threshold(trained_bigan):
    data, _, run = trained_bigan
    df = ablate_bigan(run, {"synthetic": data}, (0,), alarm_budget_per_hour=1.0,
                      half_life_min=15.0, persistence=3)
    assert sorted(df["component"]) == sorted(COMPONENTS)
    assert df["padf"].between(0, 1).all()
    # combined совпадает с обычным прогоном по тем же score
    saved = np.load(run / "scores" / "cand_gcngru_bigan__synthetic__seed0.npy")
    from stad.checkpoints import load_detector

    det, _ = load_detector(checkpoint_path(run, "cand_gcngru_bigan", "synthetic", 0))
    val = bigan_components(det, data.X_val)["combined"]
    ref = full_report(saved, data, val_scores=val, alarm_budget_per_hour=1.0,
                      half_life_min=15.0, persistence=3)
    got = df.loc[df["component"] == "combined", "padf"].iloc[0]
    assert got == pytest.approx(ref["padf"], abs=1e-6)


def test_ablate_bigan_detects_mismatched_saved_scores(trained_bigan, tmp_path):
    """Если сохранённые score не соответствуют чекпойнту, абляция падает, а не врёт."""
    import shutil

    data, _, run = trained_bigan
    bad = tmp_path / "bad"
    shutil.copytree(run, bad)
    np.save(bad / "scores" / "cand_gcngru_bigan__synthetic__seed0.npy",
            np.load(run / "scores" / "cand_gcngru_bigan__synthetic__seed0.npy") + 5.0)
    with pytest.raises(RuntimeError, match="расходится"):
        ablate_bigan(bad, {"synthetic": data}, (0,), alarm_budget_per_hour=1.0,
                     half_life_min=15.0, persistence=3)


def test_summary_verdict_rule():
    """Правило R5: критик информативен только при значимом выигрыше combined над cycle."""
    rng = np.random.default_rng(0)
    blocks = [f"f{i}|s0" for i in range(30)]
    base = rng.uniform(0.2, 0.4, 30)

    def frame(combined):
        rows = []
        for b, c, y in zip(blocks, base, combined):
            rows += [{"component": "cycle", "block": b, "padf": c, "event_recall": 0.5, "average_precision": 0.3},
                     {"component": "critic", "block": b, "padf": 0.1, "event_recall": 0.2, "average_precision": 0.2},
                     {"component": "combined", "block": b, "padf": y, "event_recall": 0.5, "average_precision": 0.3}]
        return pd.DataFrame(rows)

    _, same = summarise_ablation(frame(base + rng.normal(0, 0.002, 30)))
    _, better = summarise_ablation(frame(base + 0.1))
    assert "НЕ значимо лучше" in same and "D*=1/2" in same
    assert "значимо лучше cycle" in better and "НЕ значимо" not in better
