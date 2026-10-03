"""Фигуры отчёта. Каждая отвечает на один конкретный вопрос.

===  =========================================  ==================================================
№    Фигура                                      Вопрос, на который она отвечает
===  =========================================  ==================================================
01   Диаграмма критических различий              Какие архитектуры различаются **значимо**?
02   Операционные кривые                         Как меняется выбор при реальном FPR диспетчерской?
03   PR-кривые                                   Есть ли разделяющая способность вообще?
04   Теплокарта энкодер × голова                 Что важнее: архитектура или механизм score?
05   Панель инфляции метрик                      Можно ли верить PA-F1 из литературы?
06   Разброс по сидам                            Отличается ли разница методов от шума инициализации?
07   Парето: качество против стоимости           Что брать для edge-развёртывания?
08   Разбор одного события                       Как детектор ведёт себя на реальном ДТП?
===  =========================================  ==================================================

Ни одна фигура не использует две оси Y и ни одна не передаёт смысл
только цветом: везде есть маркеры, прямые подписи и CSV-двойник.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..metrics.stats import mean_ranks, nemenyi_cd, rank_matrix
from ..encoders import ENCODER_LABELS
from ..registry import GROUP_LABELS
from .theme import (
    CATEGORICAL,
    MARKERS,
    CATEGORICAL_SAFE3,
    DIVERGING,
    GRID,
    GROUP_STYLE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    annotate_source,
    save,
    wrap,
)


# --------------------------------------------------------------------- fig 01
def fig_critical_difference(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    higher_is_better: bool = True,
    out_dir: Path,
    tables_dir: Path,
    alpha: float = 0.05,
) -> list[Path]:
    """Диаграмма критических различий (Friedman + post-hoc Nemenyi).

    Главная фигура работы. Горизонтальная ось — средний ранг по блокам
    (блок = сид × датасет). Планка CD показывает, какая разница рангов
    статистически незначима: методы, соединённые линией, различить
    нельзя. Это защищает от вывода «наша модель лучше на 0.3%».
    """
    pivot = runs.pivot_table(index="block", columns="label", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return []

    ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
    cd = nemenyi_cd(pivot.shape[1], pivot.shape[0], alpha=alpha)

    k = len(ranks)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * k + 2.4))
    y = np.arange(k)[::-1]

    group_of = runs.drop_duplicates("label").set_index("label")["group"].to_dict()
    for yi, (label, r) in zip(y, ranks.items()):
        style = GROUP_STYLE.get(group_of.get(label, "candidate"), GROUP_STYLE["candidate"])
        ax.plot([0, r], [yi, yi], color=GRID, lw=1.0, zorder=1)
        ax.plot(r, yi, style["marker"], color=style["color"], markersize=9, zorder=3)
        ax.text(r + 0.12, yi, f"{r:.2f}", va="center", fontsize=9, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(wrap(ranks.index, 34), fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel(f"средний ранг по «{metric}» (меньше = лучше)")
    ax.set_xlim(0, float(ranks.max()) + 1.4)
    ax.grid(axis="y", visible=False)

    # планка критической разницы — внутри области данных, над верхней строкой
    best = float(ranks.min())
    y_cd = k - 0.35
    ax.set_ylim(-0.8, k - 0.1)
    ax.annotate(
        "", xy=(best, y_cd), xytext=(best + cd, y_cd),
        arrowprops=dict(arrowstyle="|-|,widthB=0.4,widthA=0.4", color=INK_SECONDARY, lw=1.2),
    )
    ax.text(best + cd / 2, y_cd - 0.22, f"CD = {cd:.2f}  (α={alpha})",
            ha="center", va="top", fontsize=9, color=INK_SECONDARY)
    ax.axvspan(best, best + cd, color=CATEGORICAL[0], alpha=0.07, zorder=0)

    handles = [
        Line2D([], [], marker=s["marker"], color=s["color"], ls="", label=s["label"])
        for g, s in GROUP_STYLE.items()
        if g in set(group_of.values())
    ]
    # легенда под осями: внутри она перекрывает нижние строки рейтинга
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.02 * k),
              ncol=min(4, len(handles)))
    ax.set_title("Какие архитектуры различаются значимо")
    annotate_source(
        fig,
        f"Блоков (сид × датасет): {pivot.shape[0]} · методов: {pivot.shape[1]} · "
        "методы внутри затенённой полосы от лучшего неразличимы (Nemenyi). "
        "Агрегация по сидам — среднее, не лучшее из N.",
    )

    table = ranks.rename("mean_rank").reset_index().assign(cd=cd, metric=metric, alpha=alpha)
    return save(fig, out_dir, "fig01_critical_difference", table=table, tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 02
def fig_operating_curves(
    curves: pd.DataFrame,
    *,
    out_dir: Path,
    tables_dir: Path,
    highlight: tuple[str, ...] = (),
    max_series: int = 6,
) -> list[Path]:
    """Две панели: event-recall и задержка против целевого FPR.

    Две панели, а не две оси Y: величины несопоставимы по масштабу
    (доля и минуты), и совмещение их на одном графике — самая
    распространённая ошибка в таких сравнениях.

    Зачем это нужно. Одно число при FPR = 1% скрывает форму
    компромисса: две модели с одинаковым recall при 1% ведут себя
    противоположно при 0.1%, а рабочая точка диспетчерской — именно там,
    потому что цена ложной тревоги на сети из сотен узлов высока.
    """
    labels = list(dict.fromkeys(curves["label"]))
    if highlight:
        labels = [l for l in labels if l in highlight] + [l for l in labels if l not in highlight]
    labels = labels[:max_series]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for i, label in enumerate(labels):
        sub = curves[curves["label"] == label].sort_values("alarm_budget_per_hour")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        mk = MARKERS[i % len(MARKERS)]
        axes[0].plot(sub["alarm_budget_per_hour"], sub["event_recall"],
                     marker=mk, color=color, label=label, markersize=6)
        axes[1].plot(sub["alarm_budget_per_hour"], sub["median_delay_min"],
                     marker=mk, color=color, label=label, markersize=6)
        # Прямые подписи ставятся только при малом числе серий: при шести
        # они сталкиваются на правом краю и делают фигуру нечитаемой.
        # Идентичность в этом случае несёт легенда плюс различие маркеров.
        if len(sub) and len(labels) <= 4:
            axes[0].annotate(
                wrap([label], 18)[0],
                (sub["alarm_budget_per_hour"].iloc[-1], sub["event_recall"].iloc[-1]),
                textcoords="offset points", xytext=(6, 0), fontsize=7.5,
                color=INK_SECONDARY, va="center",
            )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("бюджет ложных тревог, в час на всю сеть")
        ax.axvline(1.0, color=INK_MUTED, lw=0.9, ls=":")  # принятая рабочая точка
    axes[0].set_ylabel("доля обнаруженных событий")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Event-recall против бюджета тревог")
    axes[1].set_ylabel("медианная задержка, мин")
    axes[1].axhline(0.0, color=INK_MUTED, lw=0.9, ls="--")
    axes[1].set_title("Задержка относительно отчёта")
    axes[1].text(
        0.99, 0.02, "0 = момент официального отчёта; ниже = раньше",
        transform=axes[1].transAxes, fontsize=8, color=INK_MUTED, ha="right", va="bottom",
    )
    axes[0].legend(loc="lower right", fontsize=7.5, ncol=1)
    annotate_source(
        fig,
        "Пунктир — рабочая точка 1 подтверждённая ложная тревога в час на сеть. "
        "Ось в тревогах в час, а не в процентах FPR: это единица, в которой решает эксплуатант.",
    )
    return save(fig, out_dir, "fig02_operating_curves", table=curves, tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 03
def fig_pr_curves(
    pr_data: dict[str, tuple[np.ndarray, np.ndarray]],
    prevalence: float,
    *,
    out_dir: Path,
    tables_dir: Path,
    max_series: int = 6,
) -> list[Path]:
    """PR-кривые с линией случайного детектора.

    PR, а не ROC. При доле аномалий порядка процента ROC-AUC
    оптимистичен, и ROC-семейство метрик завышается при
    best-of-N отчётности уже при N≈9–11, тогда как PR-метрики
    остаются плоскими при любом N (Lyu, 2026).
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    rows = []
    for i, (label, (recall, precision)) in enumerate(list(pr_data.items())[:max_series]):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(recall, precision, color=color, label=label)
        rows.append({"label": label, "n_points": len(recall)})

    ax.axhline(prevalence, color=INK_MUTED, ls="--", lw=1.2)
    ax.text(0.02, prevalence * 1.08, f"случайный детектор (prevalence = {prevalence:.4f})",
            fontsize=8, color=INK_MUTED)
    ax.set_xlabel("recall по точкам")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, min(1.0, float(np.nanmax([p.max() for _, p in pr_data.values()], initial=0.1)) * 1.15)))
    ax.set_title("Разделяющая способность (PR, не ROC)")
    ax.legend(loc="upper right", fontsize=8)
    annotate_source(fig, "Пол PR-кривой равен доле аномалий — поэтому PR устойчив к инфляции, в отличие от ROC.")
    return save(fig, out_dir, "fig03_pr_curves", table=pd.DataFrame(rows), tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 04
def fig_encoder_head_heatmap(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    out_dir: Path,
    tables_dir: Path,
    eta2: pd.DataFrame | None = None,
) -> list[Path]:
    """Теплокарта «энкодер × голова» — ответ на главный вопрос бенчмарка.

    Одна последовательная шкала (один оттенок, светлый → тёмный):
    величина здесь — магнитуда, а не полярность, поэтому радужные и
    расходящиеся шкалы были бы ошибкой кодирования.

    Как читать. Если вариация **по строкам** (энкодеры) заметно меньше,
    чем **по столбцам** (головы), то выбор механизма score важнее выбора
    архитектуры энкодера, и диссертации следует сосредоточиться на
    механизме. Численная версия того же вывода — в подписи (eta²).
    """
    sub = runs[runs["encoder"].notna() & runs["head"].notna()]
    if sub.empty:
        return []
    piv = sub.pivot_table(index="encoder_label", columns="head_label", values=metric, aggfunc="mean")
    if piv.shape[0] < 2 or piv.shape[1] < 2:
        return []

    fig, ax = plt.subplots(figsize=(1.9 * piv.shape[1] + 3.2, 0.72 * piv.shape[0] + 2.6))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap=SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), wrap(piv.columns, 14), fontsize=9)
    ax.set_yticks(range(piv.shape[0]), wrap(piv.index, 20), fontsize=9)
    ax.grid(False)

    vals = piv.to_numpy(dtype=float)
    mid = np.nanmean(vals)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v > mid else INK_PRIMARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(metric)
    cb.outline.set_visible(False)

    note = "Вариация по столбцам против вариации по строкам отвечает, что важнее."
    if eta2 is not None and not eta2.empty:
        parts = ", ".join(f"{r.axis}: eta²={r.eta2:.3f}" for r in eta2.itertuples())
        note = f"Доля объяснённой дисперсии — {parts}. " + note
    ax.set_title("Что важнее: энкодер или механизм score")
    annotate_source(fig, note)
    return save(fig, out_dir, "fig04_encoder_head_heatmap",
                table=piv.reset_index(), tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 05
def fig_metric_inflation(
    runs: pd.DataFrame,
    *,
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Панель инфляции: PA-F1 против честного F1 при пороге по FPR.

    Это воспроизведение результата Kim et al. (2021) на собственных
    данных и одновременно валидация протокола. Ожидаемая картина:
    случайный контроль получает по PA-F1 значение, сопоставимое с
    обученными моделями, а по честной метрике падает к полу.

    Если этого не происходит — либо данные нетипичны, либо в реализации
    PA ошибка; и то и другое надо выяснить **до** интерпретации
    остальных фигур.
    """
    agg = (
        runs.groupby(["label", "group"], as_index=False)
        .agg(
            pa_f1=("pa_f1_INVALID_for_ranking", "mean"),
            honest_f1=("f1_at_fpr_threshold", "mean"),
            ap=("average_precision", "mean"),
        )
        .sort_values("pa_f1", ascending=False)
    )
    if agg.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.44 * len(agg) + 2.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(agg))[::-1]

    # левая панель: две честные/нечестные величины рядом
    h = 0.36
    axes[0].barh(y + h / 2, agg["pa_f1"], height=h, color=CATEGORICAL[7], label="PA-F1 (недопустимо)")
    axes[0].barh(y - h / 2, agg["honest_f1"], height=h, color=CATEGORICAL[0], label="F1 при пороге по FPR")
    axes[0].set_yticks(y, wrap(agg["label"], 32), fontsize=8.5)
    axes[0].set_xlabel("F1")
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Point-adjustment завышает F1")

    for yi, (_, row) in zip(y, agg.iterrows()):
        if row["group"] == "control":
            axes[0].annotate("контроль", (row["pa_f1"], yi + h / 2), xytext=(4, 0),
                             textcoords="offset points", fontsize=7.5, va="center",
                             color=CATEGORICAL[7], fontweight="bold")

    # правая панель: во сколько раз завышено
    ratio = (agg["pa_f1"] / agg["honest_f1"].clip(lower=1e-6)).clip(upper=60)
    colors = [GROUP_STYLE.get(g, GROUP_STYLE["candidate"])["color"] for g in agg["group"]]
    axes[1].barh(y, ratio, height=0.62, color=colors)
    axes[1].set_yticks(y, [""] * len(agg))
    axes[1].set_xlabel("во сколько раз PA завышает F1")
    axes[1].axvline(1.0, color=INK_MUTED, lw=1.0, ls="--")
    axes[1].grid(axis="y", visible=False)
    axes[1].set_title("Коэффициент инфляции")
    for yi, r in zip(y, ratio):
        axes[1].text(r + 0.4, yi, f"×{r:.1f}", va="center", fontsize=8, color=INK_SECONDARY)

    annotate_source(
        fig,
        "Воспроизведение Kim et al. (2021): если случайный контроль высоко по PA-F1 и низко "
        "по честной метрике — протокол работает правильно, а таблицы с PA-F1 из литературы "
        "несопоставимы.",
    )
    return save(fig, out_dir, "fig05_metric_inflation", table=agg.assign(inflation=ratio.to_numpy()),
                tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 06
def fig_seed_variance(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Разброс метрики по сидам — отличима ли разница методов от шума.

    Alves et al. (2026) на 100 прогонах показали, что изменчивость между
    прогонами и подвыборками сопоставима с разницей между методами.
    Поэтому в отчёте не бывает одного числа: только среднее с разбросом,
    и ранжирование имеет смысл лишь там, где интервалы не перекрываются.
    """
    agg = runs.groupby(["label", "group"], as_index=False).agg(
        mean=(metric, "mean"), std=(metric, "std"), n=(metric, "size")
    ).sort_values("mean", ascending=False)
    if agg.empty:
        return []

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(agg) + 2.4))
    y = np.arange(len(agg))[::-1]
    for yi, (_, row) in zip(y, agg.iterrows()):
        style = GROUP_STYLE.get(row["group"], GROUP_STYLE["candidate"])
        pts = runs.loc[runs["label"] == row["label"], metric].to_numpy(dtype=float)
        ax.plot(pts, np.full_like(pts, yi, dtype=float), style["marker"],
                color=style["color"], alpha=0.45, markersize=6, zorder=2)
        sd = 0.0 if not np.isfinite(row["std"]) else row["std"]
        ax.plot([row["mean"] - sd, row["mean"] + sd], [yi, yi],
                color=style["color"], lw=2.6, solid_capstyle="round", zorder=3)
        ax.plot(row["mean"], yi, "|", color=INK_PRIMARY, markersize=14, markeredgewidth=2, zorder=4)
        ax.text(row["mean"] + sd + 0.012, yi, f"{row['mean']:.3f} ± {sd:.3f}",
                va="center", fontsize=8, color=INK_SECONDARY)

    ax.set_yticks(y, wrap(agg["label"], 34), fontsize=8.5)
    ax.set_xlabel(f"{metric}: отдельные сиды (точки), среднее ± стд (планка)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Разница методов против шума инициализации")
    handles = [
        Line2D([], [], marker=s["marker"], color=s["color"], ls="", label=s["label"])
        for g, s in GROUP_STYLE.items() if g in set(agg["group"])
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1 - 0.015 * len(agg)),
              ncol=min(4, len(handles)))
    annotate_source(fig, f"Сидов на конфигурацию: {int(agg['n'].max())}. Отбор лучшего сида запрещён.")
    return save(fig, out_dir, "fig06_seed_variance", table=agg, tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 07
def fig_pareto(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    cost: str = "inference_ms_per_window",
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Парето «качество против стоимости вывода».

    Для диссертации важен не только максимум метрики: контур устранения
    работает в реальном времени, а DHMPN прямо позиционируется как
    решение для ресурсно-ограниченных терминальных устройств. Фигура
    показывает, какие конфигурации вообще находятся на фронте.

    Цветом кодируются **три** группы: для диаграммы рассеяния
    сравниваются все пары цветов, и только три слота палитры проходят
    проверку на разделимость при дальтонизме во всех парах.
    """
    sub = runs.groupby(["label", "group"], as_index=False).agg(
        q=(metric, "mean"), c=(cost, "mean"), params=("n_params", "mean")
    )
    sub = sub[np.isfinite(sub["c"]) & (sub["c"] > 0)]
    if sub.empty:
        return []

    # сводим к трём группам, чтобы соблюсти ограничение палитры all-pairs
    group3 = {"control": "Контроли и бейзлайны", "baseline": "Контроли и бейзлайны",
              "non_graph": "Без графа", "candidate": "Кандидаты"}
    sub["g3"] = sub["group"].map(group3).fillna("Кандидаты")
    markers = {"Контроли и бейзлайны": "s", "Без графа": "^", "Кандидаты": "o"}

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, (g, part) in enumerate(sub.groupby("g3")):
        ax.scatter(part["c"], part["q"], s=np.clip(part["params"] / 900.0, 40, 420),
                   c=CATEGORICAL_SAFE3[i % 3], marker=markers.get(g, "o"),
                   edgecolors="white", linewidths=1.6, label=g, zorder=3)

    # фронт Парето: максимум качества при не большей стоимости
    front = sub.sort_values("c")
    best, xs, ys = -np.inf, [], []
    for _, r in front.iterrows():
        if r["q"] > best:
            best = r["q"]
            xs.append(r["c"])
            ys.append(r["q"])
    ax.step(xs, ys, where="post", color=INK_MUTED, lw=1.3, ls="--", zorder=1, label="фронт Парето")

    # Подписи ставятся всем точкам (цвета диаграммы рассеяния ниже 3:1 к фону,
    # поэтому действует правило компенсации: смысл должен читаться без цвета).
    # Чтобы они не сталкивались, близкие по качеству точки разводятся по
    # вертикали, а подпись правой половины уходит влево от маркера.
    ax.set_xscale("log")
    xmin, xmax = float(sub["c"].min()), float(sub["c"].max())
    ax.set_xlim(xmin / 3.0, xmax * 6.0)
    span = max(float(sub["q"].max() - sub["q"].min()), 1e-9)
    ax.set_ylim(float(sub["q"].min()) - 0.12 * span, float(sub["q"].max()) + 0.18 * span)

    placed: list[tuple[float, float]] = []
    mid_x = (np.log10(xmin) + np.log10(xmax)) / 2.0
    for _, r in sub.sort_values("q", ascending=False).iterrows():
        right = np.log10(max(r["c"], 1e-12)) < mid_x
        dy = 7
        for px, pq in placed:
            if abs(np.log10(max(r["c"], 1e-12)) - px) < 0.35 and abs(r["q"] - pq) < 0.08 * span:
                dy = -16 if dy > 0 else 14
        ax.annotate(
            wrap([r["label"]], 22)[0], (r["c"], r["q"]), textcoords="offset points",
            xytext=(10 if right else -10, dy), fontsize=7.5, color=INK_SECONDARY,
            ha="left" if right else "right",
            va="bottom" if dy > 0 else "top",
        )
        placed.append((np.log10(max(r["c"], 1e-12)), r["q"]))

    ax.set_xlabel("время вывода на окно, мс (лог)")
    ax.set_ylabel(metric)
    ax.set_title("Качество против стоимости вывода")
    # легенда в середине слева: там единственная устойчиво пустая область
    ax.legend(loc="center left", fontsize=8)
    annotate_source(fig, "Размер точки — число параметров. Фронт Парето: что нельзя улучшить, не заплатив временем.")
    return save(fig, out_dir, "fig07_pareto", table=sub, tables_dir=tables_dir)


# --------------------------------------------------------------------- fig 08
def fig_event_timeline(
    event_traces: pd.DataFrame,
    *,
    out_dir: Path,
    tables_dir: Path,
    event_id: int | None = None,
) -> list[Path]:
    """Разбор одного события: score во времени, порог, момент отчёта.

    Количественная таблица не показывает **как** модель ошибается.
    Качественный разбор одного события показывает: вышла ли модель за
    порог до отчёта, был ли это одиночный выброс или устойчивый подъём,
    и совпадает ли её момент срабатывания с физикой распространения
    затора. Такой разбор обязателен и в статьях по FT-AED.
    """
    df = event_traces if event_id is None else event_traces[event_traces["event_id"] == event_id]
    if df.empty:
        return []
    eid = int(df["event_id"].iloc[0])
    df = df[df["event_id"] == eid]

    labels = list(dict.fromkeys(df["label"]))[:4]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for i, label in enumerate(labels):
        part = df[df["label"] == label].sort_values("minutes_from_report")
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(part["minutes_from_report"], part["score_norm"], color=color, label=label)
        thr = float(part["threshold_norm"].iloc[0])
        ax.axhline(thr, color=color, ls=":", lw=1.0, alpha=0.75)
        fires = part[part["score_norm"] > thr]
        if len(fires):
            x0 = float(fires["minutes_from_report"].iloc[0])
            ax.plot([x0], [float(fires["score_norm"].iloc[0])], "o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            ax.annotate(f"{x0:+.0f} мин", (x0, float(fires["score_norm"].iloc[0])),
                        textcoords="offset points", xytext=(8, 6), fontsize=8, color=color)

    ax.axvline(0.0, color=INK_PRIMARY, lw=1.4)
    ax.text(0.3, ax.get_ylim()[1] * 0.96, "официальный отчёт", fontsize=8,
            color=INK_PRIMARY, va="top")
    ax.set_xlabel("минуты относительно официального отчёта (отрицательные = раньше)")
    ax.set_ylabel("нормированный anomaly score")
    ax.set_title(f"Разбор события #{eid}: кто сработал раньше")
    ax.legend(loc="upper left", fontsize=8)
    annotate_source(fig, "Пунктир — порог при FPR 1% для соответствующей модели; маркер — первая сработка.")
    return save(fig, out_dir, f"fig08_event_{eid}", table=df, tables_dir=tables_dir)


# ----------------------------------------------------------------- fig 09 (доп.)
def fig_spatial_prior_contribution(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Вклад пространственного prior при выровненном бюджете.

    Отдельная фигура, потому что это утверждение из бенчмарка FT-AED,
    которое мы проверяем напрямую: «игнорирование пространственных
    связей ухудшает качество». Расходящаяся шкала уместна здесь и только
    здесь: величина знаковая (прирост или потеря относительно
    неграфового среднего), и нейтральный серый на нуле нужен.
    """
    sub = runs[runs["encoder"].notna()].copy()
    if sub.empty or "uses_graph" not in sub.columns:
        return []
    base = sub.loc[~sub["uses_graph"], metric].mean()
    if not np.isfinite(base):
        return []

    agg = sub.groupby(["encoder_label", "uses_graph"], as_index=False)[metric].mean()
    agg["delta"] = agg[metric] - base
    agg = agg.sort_values("delta")

    vmax = float(np.nanmax(np.abs(agg["delta"]))) or 1.0
    cmap = plt.get_cmap(DIVERGING)
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(agg) + 2.4))
    y = np.arange(len(agg))[::-1]
    colors = [cmap(0.5 + 0.5 * d / vmax) for d in agg["delta"]]
    ax.barh(y, agg["delta"], color=colors, height=0.6)
    ax.axvline(0.0, color=INK_PRIMARY, lw=1.2)
    ax.set_yticks(y, wrap(agg["encoder_label"], 26), fontsize=9)
    ax.set_xlabel(f"Δ {metric} относительно среднего неграфовых энкодеров")
    ax.grid(axis="y", visible=False)
    ax.set_title("Окупается ли пространственный prior")
    for yi, d in zip(y, agg["delta"]):
        ax.text(d + np.sign(d) * 0.004, yi, f"{d:+.3f}", va="center", fontsize=8,
                ha="left" if d >= 0 else "right", color=INK_SECONDARY)
    annotate_source(
        fig,
        f"База (неграфовые, среднее {metric}) = {base:.3f}. Бенчмарк FT-AED утверждает, что "
        "отказ от пространственных связей ухудшает качество — здесь это проверяется напрямую "
        "при выровненном бюджете параметров.",
    )
    return save(fig, out_dir, "fig09_spatial_prior", table=agg, tables_dir=tables_dir)


# ----------------------------------------------------------------- fig 10
def fig_augmentation_effect(
    summary: pd.DataFrame,
    *,
    metric: str = "padf",
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Эффект аугментации: Δ метрики против доли синтетики, по энкодерам.

    Малые множества — по панели на энкодер, — а не одна перегруженная
    фигура: вопрос «зависит ли эффект от архитектуры» читается именно
    из сравнения панелей между собой. Внутри панели всего две серии
    (GAN и физическая инъекция), поэтому обе получают прямые подписи, и
    идентичность не держится на одном цвете.

    Как читать. Нулевая линия — обучение без синтетики. Если обе серии
    лежат на нуле, синтетика не помогает. Если обе выше и близки друг к
    другу, помогает сам факт расширения редкого класса, а не выученное
    распределение. Если выше только GAN — нужно доказывать, что это не
    запоминание артефактов генератора.
    """
    if summary.empty:
        return []
    encoders = list(dict.fromkeys(summary["encoder"]))
    n = len(encoders)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n + 1.0, 4.4), sharey=True, squeeze=False)
    axes = axes[0]

    source_style = {
        "gan": {"color": CATEGORICAL[0], "marker": "o", "label": "GAN (WGAN-GP)"},
        "lwr": {"color": CATEGORICAL[1], "marker": "s", "label": "Физика LWR (без обучения)"},
    }

    for ax, enc in zip(axes, encoders):
        part = summary[summary["encoder"] == enc]
        ax.axhline(0.0, color=INK_PRIMARY, lw=1.2)
        for src, st in source_style.items():
            s = part[part["source"] == src].sort_values("ratio")
            if s.empty:
                continue
            ax.plot(s["ratio"], s["delta_mean"], marker=st["marker"],
                    color=st["color"], label=st["label"])
            sd = s["delta_std"].fillna(0.0).to_numpy()
            ax.fill_between(s["ratio"], s["delta_mean"] - sd, s["delta_mean"] + sd,
                            color=st["color"], alpha=0.12, linewidth=0)
            ax.annotate(st["label"].split(" ")[0],
                        (s["ratio"].iloc[-1], s["delta_mean"].iloc[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=8,
                        color=st["color"], va="center")
        ax.set_title(ENCODER_LABELS.get(enc, enc), fontsize=10)
        ax.set_xlabel("синтетических окон на одно реальное")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel(f"Δ {metric} относительно обучения без синтетики")
    axes[0].legend(loc="best", fontsize=8)
    annotate_source(
        fig,
        "Полоса — ±1 стд по фолдам и сидам. Эффект считается парно: к базе того же "
        "энкодера, фолда и сида, иначе в него попадает разброс инициализации. "
        "Оценка только на реальных событиях — синтетика в тест не попадает.",
    )
    return save(fig, out_dir, "fig10_augmentation_effect", table=summary, tables_dir=tables_dir)


# ----------------------------------------------------------------- fig 11
def fig_synthetic_fidelity(
    real_delta: np.ndarray,
    gan_delta: np.ndarray,
    lwr_delta: np.ndarray,
    feature_names: list[str],
    *,
    out_dir: Path,
    tables_dir: Path,
) -> list[Path]:
    """Похожа ли синтетика на реальные аномалии по знаку и порядку величины.

    Это не метрика качества генерации, а **санитарная проверка**: если
    синтетическое возмущение не совпадает с реальным даже по знаку, то
    любой вывод об эффекте аугментации преждевременен. Сравниваются
    распределения остатка по каждому признаку в единицах sigma.
    """
    F = len(feature_names)
    fig, axes = plt.subplots(1, F, figsize=(3.4 * F + 0.8, 4.0), squeeze=False)
    axes = axes[0]
    series = {
        "реальные": (real_delta, CATEGORICAL[2]),
        "GAN": (gan_delta, CATEGORICAL[0]),
        "физика LWR": (lwr_delta, CATEGORICAL[1]),
    }
    rows = []
    for f, (ax, name) in enumerate(zip(axes, feature_names)):
        data = [np.asarray(d)[..., f].ravel() for d, _ in series.values()]
        parts = ax.boxplot(data, tick_labels=list(series), showfliers=False,
                           widths=0.55, patch_artist=True, medianprops={"color": INK_PRIMARY})
        for patch, (_, color) in zip(parts["boxes"], series.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        ax.axhline(0.0, color=INK_MUTED, lw=1.0, ls="--")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="x", visible=False)
        for label, (d, _) in series.items():
            v = np.asarray(d)[..., f].ravel()
            rows.append({"feature": name, "source": label,
                         "mean": float(v.mean()), "std": float(v.std()),
                         "median": float(np.median(v))})
    axes[0].set_ylabel("остаток относительно нормы, sigma")
    annotate_source(
        fig,
        "Санитарная проверка, а не метрика качества генерации: совпадают ли знак и "
        "порядок величины возмущения. Пунктир — отсутствие возмущения.",
    )
    return save(fig, out_dir, "fig11_synthetic_fidelity",
                table=pd.DataFrame(rows), tables_dir=tables_dir)
