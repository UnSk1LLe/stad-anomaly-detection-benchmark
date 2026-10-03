"""Прогресс обучения: формат строки, оценка оставшегося времени, отсутствие побочных эффектов."""
from __future__ import annotations

import numpy as np
import pytest

from stad.data import make_synthetic_corridor
from stad.model import build_detector
from stad.registry import CORE
from stad.runner import estimate_remaining
from stad.train import TrainConfig, format_duration, progress_line, train_detector


def test_format_duration():
    assert format_duration(5) == "5с"
    assert format_duration(95) == "1м35с"
    assert format_duration(3700) == "1ч01м"
    assert format_duration(-3) == "0с"


def test_progress_line_reports_epoch_and_eta():
    # 10 эпох из 40 за 50 с -> 5 с/эпоху, осталось 30 эпох = 150 с = 2м30с
    line = progress_line(9, 40, 0.5, 0.6, 50.0, stale=2, patience=8)
    assert "эпоха 10/40" in line and "5.0с/эп" in line and "2м30с" in line
    assert "ранняя остановка 2/8" in line
    assert line.count("#") == 5                       # 10/40 от ширины 20


def test_estimate_remaining_uses_same_config_then_same_kind():
    deep = next(c for c in CORE if c.is_trainable and not c.randomize_only)
    base = next(c for c in CORE if not c.is_trainable)
    trainable = {deep.name: True, base.name: False}
    # обучаемая клетка 60 с, бейзлайн 1 с; впереди ещё одна обучаемая и один бейзлайн
    est = estimate_remaining([deep, base], {deep.name: 60.0, base.name: 1.0}, trainable)
    assert est == pytest.approx(61.0)
    # конфигурации, которой не было, достаётся среднее по её РОДУ, а не по всем клеткам
    other_deep = next(c for c in CORE if c.is_trainable and c.name != deep.name)
    trainable[other_deep.name] = True
    est = estimate_remaining([other_deep], {deep.name: 60.0, base.name: 1.0}, trainable)
    assert est == pytest.approx(60.0)


def test_progress_does_not_change_scores(capsys):
    """Прогресс — только вывод: обучение с ним и без него даёт те же score."""
    data = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)

    def run(progress):
        import torch

        torch.manual_seed(0)                          # начальные веса строятся до set_seed в обучении
        det = build_detector("gcn_gru", "recon", hidden=8, n_features=data.n_features,
                             n_nodes=data.n_nodes, window=data.window)
        cfg = TrainConfig(epochs=3, batch_size=64, device="cpu", progress=progress)
        return train_detector(det, data, cfg, seed=0).scores

    quiet = run(False)
    out_quiet = capsys.readouterr().out
    loud = run(True)
    out_loud = capsys.readouterr().out

    np.testing.assert_allclose(quiet, loud)
    assert "эпоха" not in out_quiet
    assert "эпоха 3/3" in out_loud                    # не tty: строка печатается на последней эпохе
