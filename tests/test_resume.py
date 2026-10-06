"""Возобновление прогона после аварийной остановки: те же числа, без переобучения."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from stad import runner
from stad.checkpoints import checkpoint_path
from stad.data import make_synthetic_corridor
from stad.registry import Config
from stad.train import TrainConfig

BUDGET = 20_000
CFG_TRAIN = Config(name="t_gcn", group="candidate", encoder="gcn_gru", head="recon", rationale="тест")
CFG_BASE = Config(name="b_snd", group="baseline", baseline="snd", rationale="тест")
CFG_PHYS = Config(name="t_phys", group="candidate", encoder="gcn_gru", head="physics", rationale="тест")
METRICS = ["padf", "event_recall", "average_precision", "threshold", "alarms_per_hour",
           "median_delay_min", "n_params", "hidden", "epochs_run", "best_val_loss"]


@pytest.fixture(scope="module")
def data():
    return make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)


@pytest.fixture
def train_calls(monkeypatch):
    calls: list[int] = []
    orig = runner.train_detector

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(runner, "train_detector", counting)
    return calls


def grid(data, out, *, configs=(CFG_TRAIN, CFG_BASE), resume=False, budget=BUDGET):
    return runner.run_grid(
        configs, {"d": data}, seeds=(0,), param_budget=budget, out_dir=out,
        train_cfg=TrainConfig(epochs=2, batch_size=64, device="cpu", patience=3),
        alarm_budget_per_hour=1.0, verbose=False, resume=resume,
    )


def test_resume_reproduces_metrics_without_retraining(data, tmp_path, train_calls):
    first = grid(data, tmp_path)
    assert len(train_calls) == 1                        # обучалась одна конфигурация
    train_calls.clear()

    again = grid(data, tmp_path, resume=True)
    assert train_calls == [], "готовая клетка переобучалась"

    a = first["runs"].set_index("config")
    b = again["runs"].set_index("config")
    pd.testing.assert_frame_equal(a[METRICS].sort_index(), b[METRICS].sort_index(), check_exact=False,
                                  rtol=1e-6, atol=1e-9)
    assert bool(b.loc["t_gcn", "resumed"]) and not bool(b.loc["b_snd", "resumed"])
    # время обучения и вывода восстанавливается из чекпойнта, а не теряется
    assert b.loc["t_gcn", "train_seconds"] == pytest.approx(a.loc["t_gcn", "train_seconds"])
    assert b.loc["t_gcn", "inference_ms_per_window"] == pytest.approx(
        a.loc["t_gcn", "inference_ms_per_window"])
    assert b.loc["t_gcn", "timing_source"] == "checkpoint"
    assert (a["timing_source"] == "measured").all() and b.loc["b_snd", "timing_source"] == "measured"
    pd.testing.assert_frame_equal(
        first["curves"].sort_values(["config", "alarm_budget_per_hour"]).reset_index(drop=True),
        again["curves"].sort_values(["config", "alarm_budget_per_hour"]).reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-9,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_resumed_cells"] == 1
    assert len(again["budget"]) == 1


def test_resume_from_checkpoint_without_timing_marks_missing(data, tmp_path, train_calls):
    """Чекпойнт, записанный до сохранения времени: NaN и явная пометка, а не тихий пропуск."""
    grid(data, tmp_path)
    ckpt = checkpoint_path(tmp_path, "t_gcn", "d", 0)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("train_seconds", "inference_ms_per_window"):
        blob["extra"].pop(key)
    torch.save(blob, ckpt)
    train_calls.clear()

    row = grid(data, tmp_path, resume=True)["runs"].set_index("config").loc["t_gcn"]
    assert train_calls == [] and bool(row["resumed"])
    assert np.isnan(row["train_seconds"]) and np.isnan(row["inference_ms_per_window"])
    assert row["timing_source"] == "missing"


def test_resume_restores_physics_extras(data, tmp_path, train_calls):
    first = grid(data, tmp_path, configs=(CFG_PHYS,))
    train_calls.clear()
    again = grid(data, tmp_path, configs=(CFG_PHYS,), resume=True)
    assert train_calls == []
    assert again["runs"]["physics_correction_share"].iloc[0] == pytest.approx(
        first["runs"]["physics_correction_share"].iloc[0], rel=1e-4)


def test_resume_refuses_changed_configuration(data, tmp_path):
    grid(data, tmp_path)
    with pytest.raises(RuntimeError, match="возобновление отклонено"):
        grid(data, tmp_path, resume=True, budget=BUDGET * 2)


def test_missing_checkpoint_means_retrain(data, tmp_path, train_calls):
    grid(data, tmp_path)
    checkpoint_path(tmp_path, "t_gcn", "d", 0).unlink()
    train_calls.clear()
    out = grid(data, tmp_path, resume=True)
    assert len(train_calls) == 1
    assert not out["runs"].set_index("config").loc["t_gcn", "resumed"]


def test_corrupt_score_file_means_retrain(data, tmp_path, train_calls):
    grid(data, tmp_path)
    (tmp_path / "scores" / "t_gcn__d__seed0.npy").write_bytes(b"not an npy file")
    train_calls.clear()
    out = grid(data, tmp_path, resume=True)
    assert len(train_calls) == 1 and len(out["runs"]) == 2


def test_resume_accepts_directory_without_fingerprint(data, tmp_path, train_calls):
    """Каталог прогона, начатого до введения отпечатка: принимается на доверии."""
    grid(data, tmp_path)
    (tmp_path / "resume_fingerprint.json").unlink()
    train_calls.clear()
    out = grid(data, tmp_path, resume=True)
    assert train_calls == [] and bool(out["runs"].set_index("config").loc["t_gcn", "resumed"])
    assert (tmp_path / "resume_fingerprint.json").exists()   # дальше уже сверяется


def test_stale_hidden_means_retrain(data, tmp_path, train_calls):
    """Чекпойнт под другой бюджет не подхватывается, даже если отпечатка нет."""
    grid(data, tmp_path)
    (tmp_path / "resume_fingerprint.json").unlink()
    train_calls.clear()
    grid(data, tmp_path, resume=True, budget=BUDGET * 3)
    assert len(train_calls) == 1
