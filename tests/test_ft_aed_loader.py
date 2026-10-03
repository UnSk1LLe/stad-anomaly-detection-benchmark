"""Загрузчик FT-AED на крошечном файле с настоящей схемой (без скачивания данных)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stad.data import load_ft_aed

N_MM, LANES, DAYS, STEPS = 4, 4, 6, 240        # 4 поста × 4 полосы, 6 дней по 2 часа (шаг 30 с)
CRASH_STEP, HUMAN_STEP = 120, 180               # crash — тестовый день 6, human — обучающий день 2


def _write_csv(root, *, crash_day=6, human_day=2):
    rng = np.random.default_rng(0)
    rows = []
    t0 = 1_696_000_000
    for d in range(1, DAYS + 1):
        for s in range(STEPS):
            for m in range(N_MM):
                row = {"day": d, "unix_time": t0 + (d - 1) * 86400 + s * 30,
                       "milemarker": 53.3 + m,
                       "human_label": int(d == human_day and s == HUMAN_STEP),
                       "crash_record": int(d == crash_day and s == CRASH_STEP)}
                for ln in range(1, LANES + 1):
                    row[f"lane{ln}_speed"] = 60 + rng.normal(0, 2)
                    row[f"lane{ln}_volume"] = 10 + rng.normal(0, 1)
                    row[f"lane{ln}_occ"] = 0.1 + rng.normal(0, 0.01)
                rows.append(row)
    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "nashville_freeway_anomaly.csv", index=False)
    assert (root / "nashville_freeway_anomaly.csv").stat().st_size > 1_000_000


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("ftaed")
    _write_csv(r)
    return r


KW = dict(resample_min=1.0, window=8, clean_buffer=4, n_lanes=LANES,
          train_days=0.5, val_days=0.34, lead_min=15.0, trail_min=20.0)


def test_other_event_type_is_excluded_from_train_and_val(root):
    """При label_source=crash окна вокруг человеческой отметки не должны быть «нормой».

    Человеческая отметка лежит в дне 2, то есть в обучающей/валидационной
    части. Очистка по ``both`` убирает окна обоих событий; ``crash`` обязан
    убрать тот же набор, а не только окна официального отчёта.
    """
    both = load_ft_aed(root, label_source="both", **KW)
    crash = load_ft_aed(root, label_source="crash", **KW)
    assert crash.meta["n_other_events_excluded"] == 1
    assert both.meta["n_other_events_excluded"] == 0
    assert crash.meta["dropped_train_windows"] == both.meta["dropped_train_windows"]
    assert len(crash.X_train) == len(both.X_train)
    assert len(crash.X_val) == len(both.X_val)


def test_labels_and_events_follow_label_source(root):
    """Метки теста и реестр определяются label_source; очистка их не меняет."""
    crash = load_ft_aed(root, label_source="crash", **KW)
    both = load_ft_aed(root, label_source="both", **KW)
    assert crash.meta["n_events_total"] == 1 and both.meta["n_events_total"] == 2
    assert crash.t_val is not None and len(crash.t_val) == len(crash.X_val)
    assert crash.t_val.max() < crash.t_test.min()
