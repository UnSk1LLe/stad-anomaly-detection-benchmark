"""Чувствительность вывода к двум решениям протокола, которые легко принять произвольно.

1. **Агрегация score по узлам** (``max`` / ``q99`` / ``q95`` / ``mean``).
   Метки FT-AED корридор-уровневые, поэтому 196 score сводятся в один, и способ
   свёртки — решение протокола, а не техническая деталь.
2. **Окно меток** ``lead/trail`` вокруг отметки официального отчёта. Оно задаёт
   долю положительных окон (prevalence) и потому AP и все event-level метрики.

Обе проверки работают на сохранённых score (``scores/*.npy`` и ``*__calib.npy``) и
не переобучают модели. Честная оговорка про окно меток: модели обучены с очисткой
по окну 15/20, а пересчитывается только оценка; это чувствительность ОЦЕНКИ к
разметке, а не обучения.

Правило выбора агрегации фиксировано в :func:`choose_reducer` ДО просмотра чисел:
критерий — положение случайного контроля, а не качество лучшей модели. Выбор по
«у кого выше padf» запрещён (подгонка).

Важная оговорка, которую отчёт печатает вместе с выбором: случайный score —
независимые одинаково распределённые величины, поэтому при пороге «по бюджету
ложных тревог» его ожидаемый padf НЕ зависит от способа свёртки по узлам. Меняется
положение контроля в таблице лишь потому, что меняется padf обученных моделей.
Диагностика :func:`invariance_diagnostic` показывает это на числах.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .data.types import SplitData
from .metrics import NODE_REDUCERS, full_report

#: Окна меток для проверки: (lead_min, trail_min). Базовое — 15/20 из configs/data/ft_aed.yaml.
LABEL_WINDOWS: tuple[tuple[float, float], ...] = ((10.0, 10.0), (15.0, 20.0), (20.0, 30.0))
BASE_WINDOW: tuple[float, float] = (15.0, 20.0)

#: Конфигурации, не считающиеся «обученными» при сравнении со случайным контролем
#: (та же трактовка, что в ``report.validate_protocol``).
_NOT_TRAINED = ("ctrl_random", "ctrl_untrained")

#: Порядок предпочтения при равенстве критерия: сначала действующая агрегация —
#: переход на другую требует, чтобы она была строго лучше по критерию.
TIE_BREAK: tuple[str, ...] = ("max", "q99", "q95", "mean")


# ------------------------------------------------------------------ окно меток
def relabel_test(data: SplitData, lead_min: float, trail_min: float) -> SplitData:
    """Пересчитать метки теста и реестр событий под другое окно ``[t-lead, t+trail]``.

    Логика совпадает с ``load_ft_aed``: окно положительно, если его конец лежит в
    ``[t_report - lead, t_report + trail]``; ``event_id`` получает первое событие,
    покрывшее окно. Только для корридор-уровневых меток (одинаковы по узлам).
    """
    if not data.meta.get("labels_are_corridor_level"):
        raise ValueError("relabel_test определён для корридор-уровневых меток (FT-AED)")
    t = np.asarray(data.t_test, dtype=float)
    y = np.zeros(len(t), dtype=np.int64)
    eid = np.full(len(t), -1, dtype=np.int64)
    for ev in data.events.itertuples():
        m = (t >= ev.t_report - lead_min) & (t <= ev.t_report + trail_min)
        y[m] = 1
        eid[m & (eid < 0)] = int(ev.event_id)
    present = np.unique(eid[eid >= 0])
    events = data.events[data.events["event_id"].isin(present)].reset_index(drop=True)
    if events.empty or y.sum() == 0:
        raise ValueError(f"при окне {lead_min}/{trail_min} в тесте не осталось событий")
    n = data.n_nodes
    return dataclasses.replace(
        data,
        y_test=np.repeat(y[:, None], n, axis=1),
        event_id_test=np.repeat(eid[:, None], n, axis=1),
        events=events,
    )


# -------------------------------------------------------------------- пересчёт
def _load(scores_dir: Path, config: str, dataset: str, seed: int):
    base = scores_dir / f"{config}__{dataset}__seed{seed}"
    f_test, f_calib = Path(f"{base}.npy"), Path(f"{base}__calib.npy")
    if not (f_test.exists() and f_calib.exists()):
        return None
    return np.load(f_test), np.load(f_calib)


def evaluate_saved(
    configs: list[str],
    datasets: dict[str, SplitData],
    seeds: tuple[int, ...],
    scores_dir: str | Path,
    *,
    reduce: str,
    alarm_budget_per_hour: float,
    half_life_min: float,
    persistence: int,
    transform: Callable[[SplitData], SplitData] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Метрики по сохранённым score: ``(строки config × dataset × seed, пропущенные)``."""
    scores_dir = Path(scores_dir)
    rows: list[dict] = []
    missing: list[str] = []
    for ds_name, data in datasets.items():
        d = transform(data) if transform else data
        for seed in seeds:
            for config in configs:
                got = _load(scores_dir, config, ds_name, seed)
                if got is None:
                    missing.append(f"{config}__{ds_name}__seed{seed}")
                    continue
                s_test, s_calib = got
                rep = full_report(
                    s_test, d, calib_scores=s_calib, alarm_budget_per_hour=alarm_budget_per_hour,
                    half_life_min=half_life_min, persistence=persistence, reduce=reduce,
                )
                rows.append({
                    "config": config, "dataset": ds_name, "seed": seed,
                    "block": f"{ds_name}|s{seed}", "reduce": reduce,
                    "prevalence": d.prevalence, "n_events": len(d.events),
                    **{k: rep[k] for k in ("padf", "event_recall", "median_delay_min",
                                           "average_precision", "alarms_per_hour")},
                })
    return pd.DataFrame(rows), missing


# ------------------------------------------------------------ выбор агрегации
@dataclasses.dataclass
class ReducerChoice:
    chosen: str
    table: pd.DataFrame
    reason: str


def aggregation_table(by_reducer: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Положение случайного контроля при каждой агрегации (строка = агрегация)."""
    out = []
    for red, runs in by_reducer.items():
        mean = runs.groupby("config")["padf"].mean()
        if "ctrl_random" not in mean.index:
            continue
        rnd = float(mean["ctrl_random"])
        trained = mean.drop(index=[c for c in _NOT_TRAINED if c in mean.index])
        beaten = trained[trained < rnd]
        out.append({
            "reduce": red,
            "random_padf": rnd,
            "n_trained": len(trained),
            "n_beaten_by_random": len(beaten),
            "random_below_all_trained": len(beaten) == 0,
            "random_rank": int((mean > rnd).sum()) + 1,
            "trained_mean_padf": float(trained.mean()),
            "trained_max_padf": float(trained.max()),
            "beaten_configs": ", ".join(beaten.index),
        })
    return pd.DataFrame(out)


def choose_reducer(by_reducer: dict[str, pd.DataFrame]) -> ReducerChoice:
    """Выбрать агрегацию ПО ПОЛОЖЕНИЮ СЛУЧАЙНОГО КОНТРОЛЯ, не по качеству моделей.

    Правило (зафиксировано до просмотра чисел):

    1. критерий — наименьшее число обученных конфигураций, которые случайный
       контроль обходит по padf (``n_beaten_by_random``);
    2. при равенстве побеждает агрегация, раньше стоящая в ``TIE_BREAK`` — то
       есть действующая ``max`` меняется только на строго лучшую по критерию;
    3. если ни одна агрегация не опускает контроль ниже всех обученных, это
       прямо сообщается: агрегация критическую проверку не чинит.
    """
    table = aggregation_table(by_reducer)
    if table.empty:
        raise ValueError("в прогоне нет ctrl_random — критерий выбора не определён")
    best = int(table["n_beaten_by_random"].min())
    tied = table[table["n_beaten_by_random"] == best]["reduce"].tolist()
    chosen = next(r for r in TIE_BREAK if r in tied)
    n_tr = int(table["n_trained"].iloc[0])
    if best == 0:
        why = (f"при {chosen} случайный контроль ниже всех {n_tr} обученных конфигураций "
               f"(критерий {tied} — равный, взят первый по порядку {TIE_BREAK}).")
    else:
        why = (f"НИ ОДНА агрегация не опускает случайный контроль ниже всех обученных: "
               f"минимум обойдённых = {best} из {n_tr} ({tied}); взят {chosen} по порядку "
               f"{TIE_BREAK}. Агрегация критическую проверку не чинит.")
    return ReducerChoice(chosen, table, why)


def invariance_diagnostic(table: pd.DataFrame) -> str:
    """Показать, что случайный контроль почти не реагирует на агрегацию, а обученные — реагируют."""
    r_spread = float(table["random_padf"].max() - table["random_padf"].min())
    m_spread = float(table["trained_mean_padf"].max() - table["trained_mean_padf"].min())
    if max(r_spread, m_spread) < 1e-9:
        verdict = ("Агрегация ничего не меняет (метки поузловые, свёртки по узлам нет): "
                   "выбор не определён данными.")
    elif m_spread > 1.5 * r_spread:
        verdict = (
            "Положение контроля меняется почти целиком за счёт обученных моделей, а не самого "
            "контроля: критерий выбора фактически эквивалентен выбору по качеству обученных, "
            "что CLAUDE.md и постановка задачи запрещают как подгонку. Решение остаётся за "
            "автором; правило выбора применено как задано."
        )
    else:
        verdict = "Случайный контроль заметно реагирует на агрегацию — критерий содержателен."
    return (f"Разброс padf случайного контроля по агрегациям: {r_spread:.4f}; "
            f"разброс среднего padf обученных: {m_spread:.4f}. {verdict}")


# ------------------------------------------------------------ окно меток: итог
def ranking_stability(by_window: dict[tuple[float, float], pd.DataFrame]) -> pd.DataFrame:
    """Как меняется ранжирование методов при смене окна меток относительно базового."""
    base = by_window[BASE_WINDOW].groupby("config")["padf"].mean()
    rows = []
    for win, runs in by_window.items():
        mean = runs.groupby("config")["padf"].mean().reindex(base.index)
        rho = float(stats.spearmanr(base, mean).statistic) if win != BASE_WINDOW else 1.0
        trained = mean.drop(index=[c for c in _NOT_TRAINED if c in mean.index])
        rnd = mean.get("ctrl_random", np.nan)
        rows.append({
            "lead_min": win[0], "trail_min": win[1],
            "prevalence": float(runs["prevalence"].mean()),
            "n_events_mean": float(runs.groupby("block")["n_events"].first().mean()),
            "spearman_vs_base": rho,
            "top3": ", ".join(mean.sort_values(ascending=False).index[:3]),
            "random_beats_n_trained": int((trained < rnd).sum()) if np.isfinite(rnd) else -1,
        })
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, digits: int = 4) -> str:
    """Markdown-таблица без зависимости от tabulate."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(
    out_dir: str | Path,
    *,
    agg: ReducerChoice | None,
    agg_runs: dict[str, pd.DataFrame] | None,
    win_summary: pd.DataFrame | None,
    win_runs: dict[tuple[float, float], pd.DataFrame] | None,
    meta: dict,
) -> Path:
    """Записать CSV и ``SENSITIVITY.md`` в ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L: list[str] = ["# Чувствительность протокола\n"]
    L.append(f"Прогон: `{meta.get('run_dir')}`, сидов {meta.get('n_seeds')}, блоков "
             f"{meta.get('n_blocks')}. Метрики пересчитаны по сохранённым score; порог "
             f"калибруется на валидации.\n")

    if agg is not None and agg_runs is not None:
        pd.concat(agg_runs.values()).to_csv(out / "aggregation_runs.csv", index=False, encoding="utf-8")
        agg.table.to_csv(out / "aggregation.csv", index=False, encoding="utf-8")
        L.append("## 1. Агрегация score по узлам\n")
        L.append("Критерий — положение случайного контроля (не качество лучшей модели).\n")
        L.append(_md(agg.table.drop(columns=["beaten_configs"])))
        L.append(f"\n**Выбрано: `{agg.chosen}`.** {agg.reason}\n")
        L.append(f"**Диагностика.** {invariance_diagnostic(agg.table)}\n")

    if win_summary is not None and win_runs is not None:
        pd.concat([r.assign(lead_min=k[0], trail_min=k[1]) for k, r in win_runs.items()]
                  ).to_csv(out / "label_window_runs.csv", index=False, encoding="utf-8")
        win_summary.to_csv(out / "label_window.csv", index=False, encoding="utf-8")
        L.append("## 2. Окно меток lead/trail\n")
        L.append(_md(win_summary, 3))
        worst = float(win_summary["spearman_vs_base"].min())
        L.append(
            "\n**Вывод.** "
            + ("Ранжирование методов устойчиво к окну меток (ρ Спирмена ≥ 0.9)."
               if worst >= 0.9 else
               f"Ранжирование МЕНЯЕТСЯ при смене окна (минимальная ρ Спирмена {worst:.2f}). "
               "Это свойство разметки, а не архитектур: окно задаёт prevalence и то, что "
               "считается обнаружением. Любое утверждение о превосходстве нужно сопровождать "
               "указанием окна меток.")
            + "\n\nОграничение: модели обучены с очисткой по окну 15/20; пересчитывалась только "
              "оценка. Набор событий берётся из реестра базового окна, поэтому события, чьё "
              "окно целиком вне теста при 15/20, не добавляются.\n"
        )
    path = out / "SENSITIVITY.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


__all__ = [
    "BASE_WINDOW", "LABEL_WINDOWS", "NODE_REDUCERS", "TIE_BREAK", "ReducerChoice",
    "aggregation_table", "choose_reducer", "evaluate_saved", "invariance_diagnostic",
    "ranking_stability", "relabel_test", "write_report",
]
