"""Загрузчик FT-AED на крошечном файле с настоящей схемой (без скачивания данных).

Раскладка дней (6 дней по 4 часа, шаг 30 с, после агрегации 1 мин — 240 окон в день):
дни 1-3 — train, дни 4-5 — валидация, день 6 — тест. События:

* crash:  день 6, шаг 360  (оцениваемое, в тесте);
* human:  день 2, шаг 360  (в train);
* human:  день 5, шаг 240  (в валидации);
* human:  день 6, шаг 80   (в тесте, не пересекается с crash).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stad.data import load_ft_aed

N_MM, LANES, DAYS, STEPS = 4, 4, 6, 480
MARKS = {(6, 360): "crash", (2, 360): "human", (5, 240): "human", (6, 80): "human"}
PER_DAY = STEPS // 2                       # окон в сутки после агрегации до 1 мин


def minute(day: int, raw_step: int) -> float:
    """Время отметки в минутах от начала записи — шкала t_test/t_calib."""
    return (day - 1) * PER_DAY + raw_step // 2


def _write_csv(root):
    rng = np.random.default_rng(0)
    rows = []
    t0 = 1_696_000_000
    for d in range(1, DAYS + 1):
        for s in range(STEPS):
            for m in range(N_MM):
                kind = MARKS.get((d, s))
                row = {"day": d, "unix_time": t0 + (d - 1) * 86400 + s * 30,
                       "milemarker": 53.3 + m,
                       "human_label": int(kind == "human"),
                       "crash_record": int(kind == "crash")}
                for ln in range(1, LANES + 1):
                    row[f"lane{ln}_speed"] = 60 + rng.normal(0, 2)
                    row[f"lane{ln}_volume"] = 10 + rng.normal(0, 1)
                    row[f"lane{ln}_occ"] = 0.1 + rng.normal(0, 0.01)
                rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "nashville_freeway_anomaly.csv", index=False)
    assert (root / "nashville_freeway_anomaly.csv").stat().st_size > 1_000_000


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("ftaed")
    _write_csv(r)
    return r


KW = dict(resample_min=1.0, window=8, clean_buffer=4, n_lanes=LANES,
          train_days=0.5, val_days=0.34, lead_min=15.0, trail_min=20.0)
LEAD, TRAIL = KW["lead_min"], KW["trail_min"]


@pytest.fixture(scope="module")
def crash(root):
    return load_ft_aed(root, label_source="crash", **KW)


@pytest.fixture(scope="module")
def both(root):
    return load_ft_aed(root, label_source="both", **KW)


def near(t: np.ndarray, center: float) -> np.ndarray:
    return (t >= center - LEAD) & (t <= center + TRAIL)


def test_other_event_type_is_excluded_from_train_and_val(crash, both):
    """При label_source=crash окна вокруг человеческих отметок не должны быть «нормой».

    Очистка по ``both`` убирает окна всех событий; ``crash`` обязан убрать
    тот же набор из train и val, а не только окна официального отчёта.
    """
    assert crash.meta["n_other_events_excluded"] == 3
    assert both.meta["n_other_events_excluded"] == 0
    assert crash.meta["dropped_train_windows"] == both.meta["dropped_train_windows"]
    assert len(crash.X_train) == len(both.X_train)
    assert len(crash.X_val) == len(both.X_val)


def test_labels_and_events_follow_label_source(crash, both):
    assert crash.meta["n_events_total"] == 1 and both.meta["n_events_total"] == 4
    assert crash.t_val is not None and len(crash.t_val) == len(crash.X_val)
    assert crash.t_val.max() < crash.t_test.min()


def test_calibration_pool_keeps_buffers_but_not_label_windows(crash, both):
    """Калибровка видит буферы вокруг событий и не видит размеченные окна.

    Тестовые «нормальные» окна содержат буферные зоны; чистая норма давала бы
    на тесте тревог в разы больше бюджета. Окна самих событий в калибровку
    не попадают — иначе порог подгонялся бы под аномалии.
    """
    t = crash.t_calib
    assert t.max() < crash.t_test.min()
    assert len(crash.X_calib) > len(crash.X_val), "буферы вокруг событий вырезаны, как в X_val"

    human_val = minute(5, 240)
    assert not near(t, human_val).any(), "окна человеческой отметки (другой тип) попали в калибровку"
    # окна сразу за правой границей размеченного окна — буферная зона — присутствуют
    assert ((t > human_val + TRAIL) & (t <= human_val + TRAIL + KW["clean_buffer"])).any()

    # маскирование другого типа даёт ту же выборку, что и метки both
    np.testing.assert_array_equal(crash.t_calib, both.t_calib)


def test_test_ignores_windows_near_other_event_type(crash, both):
    """Окна рядом с человеческой отметкой в тесте при crash исключены: это не норма и не событие."""
    human_test, crash_ev = minute(6, 80), minute(6, 360)
    assert not near(crash.t_test, human_test).any()
    assert near(both.t_test, human_test).any()          # при both это настоящие положительные окна
    assert near(crash.t_test, crash_ev).any()           # оцениваемое событие на месте
    assert crash.meta["n_test_windows_ignored"] > 0 and both.meta["n_test_windows_ignored"] == 0
    assert len(crash.X_test) == len(both.X_test) - crash.meta["n_test_windows_ignored"]
    # метки и события оцениваемого типа не изменились
    assert len(crash.events) == 1 and crash.y_test.sum() > 0
