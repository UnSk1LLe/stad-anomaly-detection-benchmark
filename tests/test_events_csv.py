"""Раннер пишет events.csv: слагаемые padf каждой клетки, и при возобновлении те же."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stad import runner
from stad.data import make_synthetic_corridor
from stad.registry import Config
from stad.train import TrainConfig

COLUMNS = ["config", "dataset", "seed", "event_id", "t_report", "detected", "delay_min", "credit"]
CONFIGS = (
    Config(name="b_random", group="control", baseline="random", rationale="тест"),
    Config(name="b_snd", group="baseline", baseline="snd", rationale="тест"),
    Config(name="t_gcn", group="candidate", encoder="gcn_gru", head="recon", rationale="тест"),
)
CELL = ["config", "dataset", "seed"]


@pytest.fixture(scope="module")
def data():
    # 10 событий в тесте на клетку: случайный контроль находит часть из них, иначе
    # «среднее кредита == padf» не отличить от записи padf в каждую строку
    return make_synthetic_corridor(stations=4, lanes=2, days=6, step_min=2.0,
                                   n_events=40, window=8, seed=2)


def grid(data, out, *, configs=CONFIGS, seeds=(0, 1), resume=False):
    return runner.run_grid(
        configs, {"d": data}, seeds=seeds, param_budget=20_000, out_dir=out,
        train_cfg=TrainConfig(epochs=1, batch_size=64, device="cpu", patience=3),
        alarm_budget_per_hour=1.0, verbose=False, resume=resume,
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(CELL + ["event_id"]).reset_index(drop=True)


def test_events_csv_sums_to_padf_and_survives_resume(data, tmp_path):
    first = grid(data, tmp_path)
    assert "events" in first
    path = tmp_path / "events.csv"
    assert path.exists()
    ev = pd.read_csv(path, encoding="utf-8")
    assert list(ev.columns) == COLUMNS
    assert list(first["events"].columns) == COLUMNS

    runs = pd.read_csv(tmp_path / "runs.csv", encoding="utf-8")
    assert len(runs) == len(CONFIGS) * 2
    assert len(data.events) > 1
    for _, row in runs.iterrows():
        cell = ev[(ev["config"] == row["config"]) & (ev["dataset"] == row["dataset"])
                  & (ev["seed"] == row["seed"])].reset_index(drop=True)
        assert len(cell) == row["n_events"]
        assert int(cell["detected"].sum()) == row["n_detected"]
        # приёмка ТЗ v2, задача 1.2: среднее кредита по событиям клетки — её padf
        assert cell["credit"].sum() / len(cell) == pytest.approx(row["padf"], abs=1e-12)
        # строки — события теста в порядке реестра, кредит каждой — от её задержки
        assert cell["event_id"].tolist() == data.events["event_id"].tolist()
        np.testing.assert_allclose(cell["t_report"], data.events["t_report"].to_numpy(dtype=float))
        hit = cell["detected"].to_numpy()
        np.testing.assert_allclose(cell.loc[hit, "credit"],
                                   0.5 ** (np.maximum(cell.loc[hit, "delay_min"], 0.0) / 15.0))
        miss = cell[~cell["detected"]]
        assert miss["delay_min"].isna().all() and (miss["credit"] == 0).all()
    # в какой-то клетке найдена лишь часть событий: правило для пропущенных проверено не впустую
    mixed = ev.groupby(CELL)["detected"].agg(lambda d: 0 < d.sum() < len(d))
    assert mixed.any()

    again = grid(data, tmp_path, resume=True)
    assert again["runs"].set_index(["config", "seed"])["resumed"].loc["t_gcn"].all()
    ev2 = pd.read_csv(path, encoding="utf-8")
    pd.testing.assert_frame_equal(_sorted(ev), _sorted(ev2))


def test_failed_cell_has_no_events(data, tmp_path, monkeypatch):
    orig = runner.build_baseline

    def broken(name, **kw):
        if name == "snd":
            raise ValueError("сломано для теста")
        return orig(name, **kw)

    monkeypatch.setattr(runner, "build_baseline", broken)
    out = grid(data, tmp_path, configs=CONFIGS[:2])
    ev = pd.read_csv(tmp_path / "events.csv", encoding="utf-8")
    assert set(ev["config"]) == {"b_random"}
    assert len(out["failures"]) == 2


@pytest.mark.parametrize("resume", [False, True])
def test_event_table_failure_leaves_no_row(data, tmp_path, monkeypatch, resume):
    """Сбой пособытийной таблицы после готовой строки: клетка — в failures, ни строки, ни событий.

    С ``resume`` первый прогон проходит, и обучаемая клетка восстанавливается (базовые
    линии не восстанавливаются, их считать дёшево): сбой на восстановленной клетке ведёт
    к переобучению, а не роняет сетку.
    """
    configs = (CONFIGS[0], CONFIGS[2])
    if resume:
        grid(data, tmp_path, configs=configs, seeds=(0,))
    orig = runner._cell_events

    def broken(row, scores, d, **kw):
        if row["config"] == "t_gcn" and row["seed"] == 0:
            raise RuntimeError("сломано для теста")
        return orig(row, scores, d, **kw)

    monkeypatch.setattr(runner, "_cell_events", broken)
    out = grid(data, tmp_path, configs=configs, seeds=(0,), resume=resume)
    runs = pd.read_csv(tmp_path / "runs.csv", encoding="utf-8")
    ev = pd.read_csv(tmp_path / "events.csv", encoding="utf-8")
    assert out["failures"][["config", "seed"]].values.tolist() == [["t_gcn", 0]]
    assert runs["config"].tolist() == ["b_random"] and set(ev["config"]) == {"b_random"}
