"""Загрузчик FT-AED — основного бенчмарка с реальными метками.

FT-AED (Coursey et al., NeurIPS 2024 Datasets & Benchmarks): I-24 в сторону
Нэшвилла, ~18 миль, 49 постов × 4 полосы = 196 узлов, радарные детекторы,
шаг 30 с, 20 будних дней октября 2023, 940 800 строк.

Фактическая схема файла (проверена на скачанных данных, не угадана)::

    day          int    порядковый день месяца (20 будних дней октября 2023)
    unix_time    int    отметка времени, шаг 30 с
    milemarker   float  49 значений от 53.3 до 70.1
    laneN_speed  float  скорость по полосе N = 1..4
    laneN_volume float  объём
    laneN_occ    float  занятость
    human_label  int    ручная разметка аномалии
    crash_record int    официальный отчёт о ДТП

Два свойства меток определяют весь протокол и должны быть явно поняты,
прежде чем смотреть на любые числа.

**Метки корридор-уровневые.** Каждое событие помечено сразу на всех 49
постах: счётчики кратны 49 (2156 = 44 × 49 для ``crash_record``,
931 = 19 × 49 для ``human_label``). Пространственной локализации в
метках нет. Следствие: поточечные метрики по узлам измеряют не то, чем
кажутся, и основой служат event-level метрики, где score агрегируется
по сети максимумом.

**Метки точечные во времени.** Длительность каждого эпизода — один шаг
30 с. Это не продолжительность инцидента, а **отметка времени
официального сообщения**. Именно относительно неё бенчмарк измеряет
сокращение задержки («−10 мин при обнаружении 75% ДТП»), и именно
поэтому вокруг отметки строится окно ``[t − lead, t + trail]``: сигнал
в данных появляется раньше сообщения, и без опережающего окна детектор
штрафуется ровно за то, что является целью работы.

Данные не входят в репозиторий (лицензия и размер):
``python scripts/download_data.py --dataset ft-aed``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .splits import Standardizer, drop_anomalous_windows
from .types import SplitData

SOURCE = "https://github.com/acoursey3/freeway-anomaly-data"
PAPER = "https://arxiv.org/abs/2406.15283"
MEDIA = (
    "https://media.githubusercontent.com/media/acoursey3/"
    "freeway-anomaly-data/main/nashville_freeway_anomaly.csv"
)

FEATURES = ["speed", "occupancy", "volume"]
RAW_STEP_S = 30.0

EXPECTED = """
Ожидаемые столбцы (как в официальном nashville_freeway_anomaly.csv):

  day, unix_time, milemarker,
  lane1_speed, lane1_volume, lane1_occ, ... lane4_*,
  human_label, crash_record

Если имена отличаются, пропишите соответствие в configs/data/ft_aed.yaml
(ключ `columns`) — не правьте этот модуль.
"""


class DataNotDownloaded(FileNotFoundError):
    """Поднимается, когда файл FT-AED отсутствует."""


def _find_csv(root: Path) -> Path:
    candidates = sorted(
        [p for p in root.rglob("*.csv") if p.is_file() and p.stat().st_size > 1_000_000]
    )
    if not candidates:
        raise DataNotDownloaded(
            f"FT-AED не найден в {root}.\n"
            f"Скачайте: python scripts/download_data.py --dataset ft-aed\n"
            f"Прямая ссылка: {MEDIA}\nСтатья: {PAPER}\n{EXPECTED}"
        )
    named = [p for p in candidates if "anomaly" in p.name.lower()]
    return named[0] if named else candidates[0]


def _to_panel(df: pd.DataFrame, n_lanes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Широкий формат → тензоры ``[T, N, F]`` плюс оси времени, дней и меток.

    Узел нумеруется как ``mm_index * n_lanes + (lane - 1)``: соседние
    узлы — это соседние полосы одного поста, а узлы через ``n_lanes`` —
    тот же ряд на следующем посту. Такой порядок нужен и графу
    (:func:`corridor_adjacency`), и physics-голове, которая берёт
    производную по пространству шагом ``n_lanes``.
    """
    mm = np.sort(df["milemarker"].unique())
    mm_index = {v: i for i, v in enumerate(mm)}
    times = np.sort(df["unix_time"].unique())
    t_index = {v: i for i, v in enumerate(times)}

    T, M = len(times), len(mm)
    N = M * n_lanes
    panel = np.full((T, N, len(FEATURES)), np.nan, dtype=np.float32)

    ti = df["unix_time"].map(t_index).to_numpy()
    mi = df["milemarker"].map(mm_index).to_numpy()

    for lane in range(1, n_lanes + 1):
        node = mi * n_lanes + (lane - 1)
        for f, feat in enumerate(FEATURES):
            col = {"speed": f"lane{lane}_speed", "occupancy": f"lane{lane}_occ",
                   "volume": f"lane{lane}_volume"}[feat]
            panel[ti, node, f] = df[col].to_numpy(dtype=np.float32)

    # метки корридор-уровневые: один флаг на отметку времени
    lab = df.groupby("unix_time")[["human_label", "crash_record"]].max()
    lab = lab.reindex(times).fillna(0).astype(np.int8)
    day = df.groupby("unix_time")["day"].first().reindex(times).to_numpy()

    return panel, times.astype(np.int64), day.astype(np.int64), lab["crash_record"].to_numpy(), lab["human_label"].to_numpy()


def _resample(
    panel: np.ndarray, times: np.ndarray, day: np.ndarray,
    crash: np.ndarray, human: np.ndarray, step_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Агрегировать до шага ``step_min``: признаки средним, метки максимумом.

    Метка берётся максимумом, иначе точечная отметка сообщения просто
    исчезнет при усреднении, и событий в датасете не останется.
    """
    factor = int(round(step_min * 60.0 / RAW_STEP_S))
    if factor <= 1:
        return panel, times, day, crash, human

    T = (len(times) // factor) * factor
    sl = slice(0, T)
    shape = (T // factor, factor)

    panel_r = panel[sl].reshape(shape[0], factor, *panel.shape[1:]).mean(axis=1)
    times_r = times[sl].reshape(shape)[:, -1]          # конец интервала
    day_r = day[sl].reshape(shape)[:, 0]
    crash_r = crash[sl].reshape(shape).max(axis=1)
    human_r = human[sl].reshape(shape).max(axis=1)
    return panel_r.astype(np.float32), times_r, day_r, crash_r, human_r


def corridor_adjacency(n_milemarkers: int, n_lanes: int) -> np.ndarray:
    """Смежность «вдоль коридора + между смежными полосами», row-normalised.

    Связи задаются геометрией дороги, а не корреляцией данных: обучаемый
    граф — это уже вариант архитектуры (``encoders/graph.py``), и
    смешивать его с базовой структурой нельзя, иначе сравнение
    «фиксированный граф против обучаемого» теряет смысл.
    """
    N = n_milemarkers * n_lanes
    A = np.zeros((N, N), dtype=np.float32)
    for m in range(n_milemarkers):
        for l in range(n_lanes):
            i = m * n_lanes + l
            if l + 1 < n_lanes:
                A[i, i + 1] = A[i + 1, i] = 1.0          # соседняя полоса
            if m + 1 < n_milemarkers:
                j = (m + 1) * n_lanes + l
                A[i, j] = A[j, i] = 1.0                  # соседний пост, тот же ряд
    A += np.eye(N, dtype=np.float32)
    return (A / A.sum(1, keepdims=True)).astype(np.float32)


def _day_windows(day: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Индексы концов окон, целиком лежащих внутри одного дня.

    Между днями в данных разрыв в 16 часов. Окно, пересекающее границу,
    склеивает вечер одного дня с утром другого и порождает ложную
    аномалию у каждой модели сразу.
    """
    ends: list[int] = []
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        start, stop = idx.min(), idx.max()
        ends.extend(range(start + window - 1, stop + 1, stride))
    return np.array(sorted(ends), dtype=np.int64)


def _build_events(
    labels: np.ndarray, times: np.ndarray, merge_gap_min: float, step_min: float
) -> pd.DataFrame:
    """Склеить соседние отметки в события и собрать реестр.

    Отметки, стоящие ближе ``merge_gap_min`` друг к другу, считаются
    одним событием: иначе одно ДТП, помеченное двумя соседними
    интервалами, учитывалось бы дважды и завышало знаменатель recall.
    """
    marks = np.where(labels > 0)[0]
    if marks.size == 0:
        return pd.DataFrame(columns=["event_id", "t_report", "idx_report", "unix_time", "n_marks"])

    gap = max(1, int(round(merge_gap_min / step_min)))
    groups: list[list[int]] = [[int(marks[0])]]
    for i in marks[1:]:
        if i - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])

    rows = []
    for eid, g in enumerate(groups):
        first = g[0]
        rows.append({
            "event_id": eid,
            "idx_report": first,
            "unix_time": int(times[first]),
            "t_report": float(first) * step_min,     # минуты от начала записи
            "n_marks": len(g),
        })
    return pd.DataFrame(rows)


def load_ft_aed(
    root: str | Path = "data/ft-aed",
    *,
    columns: dict[str, str] | None = None,
    label_source: str = "both",
    resample_min: float = 1.0,
    window: int = 16,
    stride: int = 1,
    lead_min: float = 15.0,
    trail_min: float = 20.0,
    merge_gap_min: float = 10.0,
    train_days: float = 0.6,
    val_days: float = 0.15,
    clean_buffer: int = 8,
    n_lanes: int = 4,
    fold: int | None = None,
    n_folds: int = 4,
    fold_test_days: int = 3,
    fold_val_days: int = 2,
    fold_min_train_days: int = 6,
) -> SplitData:
    """Собрать :class:`SplitData` из реального файла FT-AED.

    Parameters
    ----------
    label_source:
        ``"crash"`` — только официальные отчёты о ДТП (44 события),
        ``"human"`` — только ручная разметка прочих аномалий (19),
        ``"both"`` — объединение (63). По умолчанию объединение: в
        статье метки описаны как «официальные отчёты плюс ручная
        разметка всех прочих потенциальных аномалий», то есть обе
        категории вместе и составляют разметку датасета.
    resample_min:
        Шаг агрегации. Исходный — 30 с. Решение влияет на главную
        метрику (задержку обнаружения), поэтому фиксируется в конфиге и
        указывается в тексте работы.
    lead_min, trail_min:
        Окно вокруг отметки сообщения, считающееся положительным.
        ``lead_min`` критичен: сигнал появляется раньше сообщения, и без
        опережающего окна раннее обнаружение засчитывается как ложная
        тревога.
    train_days, val_days:
        Доли **дней** (не строк). Сплит по дням, а не по индексу: внутри
        дня соседние окна перекрываются, и разрез посреди дня оставил бы
        почти идентичные окна по обе стороны границы. Игнорируются, если
        задан ``fold``.
    fold:
        Номер фолда кросс-валидации по дням (0 … ``n_folds``-1). Схема —
        **rolling origin**: обучение только на днях, предшествующих
        тестовым. Это строже блочной CV, где модель видела бы будущее,
        и соответствует тому, как система работала бы в эксплуатации.

        Зачем CV вообще. При одиночном сплите в тесте остаётся 18 событий
        из 63, и recall квантуется шагом 1/18 — различие между моделями
        тонет в этой зернистости. Четыре фолда по 3 тестовых дня дают
        четыре блока для теста Фридмана на сид вместо одного. Шесть
        фолдов по 2 дня проверены и отвергнуты: при ``label_source=crash``
        в фолде 0 остаётся 3 события (порог 5, ``scripts/check_folds.py``).
        Дней ровно ``fold_min_train_days + fold_val_days + n_folds *
        fold_test_days = 6 + 2 + 12 = 20``: запаса нет.
    """
    root = Path(root)
    path = _find_csv(root)
    df = pd.read_csv(path)
    if columns:
        df = df.rename(columns=columns)

    needed = {"day", "unix_time", "milemarker", "human_label", "crash_record"}
    needed |= {f"lane{l}_{s}" for l in range(1, n_lanes + 1) for s in ("speed", "volume", "occ")}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"в {path.name} нет столбцов {sorted(missing)}.\n{EXPECTED}")

    panel, times, day, crash, human = _to_panel(df, n_lanes)
    n_mm = panel.shape[1] // n_lanes

    labels = {
        "crash": crash,
        "human": human,
        "both": np.maximum(crash, human),
    }[label_source]

    panel, times, day, crash, human = _resample(panel, times, day, crash, human, resample_min)
    labels = {"crash": crash, "human": human, "both": np.maximum(crash, human)}[label_source]
    step_min = float(resample_min) if resample_min else RAW_STEP_S / 60.0

    # пропуски: заполняем вперёд по времени, остатки — медианой признака
    for f in range(panel.shape[-1]):
        col = panel[:, :, f]
        bad = ~np.isfinite(col)
        if bad.any():
            idx = np.where(~bad, np.arange(col.shape[0])[:, None], 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            col = col[idx, np.arange(col.shape[1])[None, :]]
            med = np.nanmedian(panel[:, :, f])
            panel[:, :, f] = np.nan_to_num(col, nan=0.0 if not np.isfinite(med) else med)

    events = _build_events(labels, times, merge_gap_min, step_min)
    if events.empty:
        raise ValueError(f"при label_source={label_source!r} в данных нет размеченных событий")

    # ------------------------------------------------------------ окна
    ends = _day_windows(day, window, stride)
    starts = ends - window + 1
    X = np.stack([panel[s : e + 1] for s, e in zip(starts, ends)])      # [n, T, N, F]
    X = np.transpose(X, (0, 2, 1, 3)).astype(np.float32)                # [n, N, T, F]
    t_end = ends.astype(np.float64) * step_min
    win_day = day[ends]

    # ---------------------------------------------- метки окон (корридор)
    n_win, N = len(ends), panel.shape[1]
    y = np.zeros(n_win, dtype=np.int64)
    eid = np.full(n_win, -1, dtype=np.int64)
    for ev in events.itertuples():
        m = (t_end >= ev.t_report - lead_min) & (t_end <= ev.t_report + trail_min)
        y[m] = 1
        fresh = m & (eid < 0)
        eid[fresh] = ev.event_id
    # метки не локализованы по пространству — транслируем на все узлы
    y_nodes = np.repeat(y[:, None], N, axis=1)
    eid_nodes = np.repeat(eid[:, None], N, axis=1)

    # События ДРУГОГО типа (при label_source="crash" — человеческие отметки, и
    # наоборот) оцениваться не будут, но это реальные аномалии. Считать их нормой
    # нельзя: протокол требует убрать из train и val окна ВСЕХ известных событий
    # (docs/PROTOCOL.md §6). Иначе они загрязняют модель нормы и задирают порог,
    # калиброванный на валидации (измерено: весь верхний 1% score валидации стоял
    # в пределах часа от человеческой отметки, recall обнулялся).
    other_labels = {"crash": human, "human": crash, "both": np.zeros_like(crash)}[label_source]
    y_other = np.zeros(n_win, dtype=np.int64)
    other_events = _build_events(other_labels, times, merge_gap_min, step_min)
    for ev in other_events.itertuples():
        y_other[(t_end >= ev.t_report - lead_min) & (t_end <= ev.t_report + trail_min)] = 1
    y_clean_nodes = np.repeat((y | y_other)[:, None], N, axis=1)   # только для очистки train/val

    # ------------------------------------------------- сплит по дням
    days_sorted = np.unique(day)
    if fold is None:
        n_tr = max(1, int(len(days_sorted) * train_days))
        n_va = max(1, int(len(days_sorted) * val_days))
        d_tr = set(days_sorted[:n_tr].tolist())
        d_va = set(days_sorted[n_tr : n_tr + n_va].tolist())
        d_te = set(days_sorted[n_tr + n_va :].tolist())
        split_scheme = "single-holdout"
    else:
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold={fold} вне диапазона 0..{n_folds - 1}")
        need = fold_min_train_days + fold_val_days + n_folds * fold_test_days
        if len(days_sorted) < need:
            raise ValueError(
                f"дней {len(days_sorted)}, для {n_folds} фолдов нужно >= {need}"
            )
        te_start = fold_min_train_days + fold_val_days + fold * fold_test_days
        te_stop = te_start + fold_test_days
        d_te = set(days_sorted[te_start:te_stop].tolist())
        d_va = set(days_sorted[te_start - fold_val_days : te_start].tolist())
        d_tr = set(days_sorted[: te_start - fold_val_days].tolist())   # только прошлое
        split_scheme = f"rolling-origin fold {fold}/{n_folds}"
    if not d_te:
        raise ValueError("после сплита по дням тест пуст — уменьшите train_days/val_days")

    m_tr = np.isin(win_day, list(d_tr))
    m_va = np.isin(win_day, list(d_va))
    m_te = np.isin(win_day, list(d_te))

    keep_tr = np.where(m_tr)[0][drop_anomalous_windows(X[m_tr], y_clean_nodes[m_tr], buffer=clean_buffer)]
    keep_va = np.where(m_va)[0][drop_anomalous_windows(X[m_va], y_clean_nodes[m_va], buffer=clean_buffer)]
    idx_te = np.where(m_te)[0]
    if len(keep_tr) < 64:
        raise ValueError(f"в train осталось {len(keep_tr)} окон — уменьшите clean_buffer")

    scaler = Standardizer().fit(X[keep_tr])
    # аномальные окна обучающих дней не выбрасываем: они нужны supervised-арму
    anom_tr = np.where(m_tr)[0][y_nodes[m_tr][:, 0] == 1]
    anom_va = np.where(m_va)[0][y_nodes[m_va][:, 0] == 1]
    ev_test = events[events["event_id"].isin(np.unique(eid_nodes[idx_te][eid_nodes[idx_te] >= 0]))]
    ev_test = ev_test.reset_index(drop=True)
    if ev_test.empty:
        raise ValueError("в тестовых днях нет ни одного события — сдвиньте границу сплита")

    data = SplitData(
        X_train=scaler.transform(X[keep_tr]),
        X_val=scaler.transform(X[keep_va]) if len(keep_va) else scaler.transform(X[keep_tr][:16]),
        X_test=scaler.transform(X[idx_te]),
        y_test=y_nodes[idx_te],
        event_id_test=eid_nodes[idx_te],
        t_test=t_end[idx_te],
        events=ev_test,
        A=corridor_adjacency(n_mm, n_lanes),
        feature_names=FEATURES,
        X_train_anomalous=scaler.transform(X[anom_tr]) if len(anom_tr) else None,
        X_val_anomalous=scaler.transform(X[anom_va]) if len(anom_va) else None,
        # без валидации X_val подменён куском train, и калибровать порог на нём
        # нельзя: t_val=None заставляет калибровку упасть, а не молча сработать
        t_val=t_end[keep_va] if len(keep_va) else None,
        meta={
            "dataset": "FT-AED",
            "source": SOURCE,
            "paper": PAPER,
            "file": path.name,
            "n_nodes": int(N),
            "n_milemarkers": int(n_mm),
            "lanes": int(n_lanes),
            "step_min": step_min,
            "window": int(window),
            "stride": int(stride),
            "label_source": label_source,
            "split_scheme": split_scheme,
            "fold": fold,
            "lead_min": lead_min,
            "trail_min": trail_min,
            "n_events_total": int(len(events)),
            "n_events_test": int(len(ev_test)),
            "days_train": sorted(d_tr),
            "days_val": sorted(d_va),
            "days_test": sorted(d_te),
            "n_train_windows": int(len(keep_tr)),
            "n_val_windows": int(len(keep_va)),
            "n_test_windows": int(len(idx_te)),
            "dropped_train_windows": int(m_tr.sum() - len(keep_tr)),
            "n_other_events_excluded": int(len(other_events)),
            "n_train_anomalous": int(len(anom_tr)),
            "n_val_anomalous": int(len(anom_va)),
            "labels_are_corridor_level": True,
            "label_note": (
                "Метки не локализованы по пространству: отметка события транслирована "
                "на все узлы. Поточечные метрики по узлам поэтому справочные, "
                "ранжирование ведётся по event-level."
            ),
        },
    )
    data.validate()
    return data
