"""Интерпретация результатов и вывод для диссертации.

Это не «генератор текста по шаблону». Модуль реализует явные решающие
правила, записанные **до** прогона эксперимента (см. docs/DECISION_RULES.md),
и применяет их к полученным числам. Смысл в том, чтобы вывод нельзя было
подогнать под результат: правила фиксированы, меняются только данные.

Порядок строгий:

1. **Валидация протокола.** Если случайный контроль ведёт себя не так,
   как предсказывает теория, остальные выводы не делаются вообще.
2. **Статистика.** Friedman по блокам, затем CD и парные bootstrap-тесты
   относительно референсной конфигурации.
3. **Разложение дисперсии.** Что объясняет больше разброса — энкодер
   или механизм score.
4. **Решающие правила → рекомендация по главам диссертации.**
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics.stats import (
    friedman,
    mean_ranks,
    nemenyi_cd,
    paired_bootstrap_test,
    variance_decomposition,
)
from .registry import EXTENDED, PRIMARY_COMPARISON, REFERENCE


@dataclass
class Check:
    """Один пункт валидации протокола."""

    name: str
    passed: bool
    detail: str
    critical: bool = True

    @property
    def mark(self) -> str:
        return "OK" if self.passed else ("ПРОВАЛ" if self.critical else "ВНИМАНИЕ")


def primary_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Прогоны предзарегистрированного подмножества для выводной статистики.

    Возвращает ``(подмножество, примечание)``. Если в прогоне нет хотя бы
    трёх конфигураций из ``PRIMARY_COMPARISON`` (smoke, синтетика), тест
    считается по всем конфигурациям, и примечание об этом говорит прямо:
    итоговая сетка ``core`` содержит подмножество целиком.
    """
    present = [c for c in PRIMARY_COMPARISON if c in set(runs["config"])]
    if len(present) >= 3:
        missing = [c for c in PRIMARY_COMPARISON if c not in present]
        note = f"отсутствуют в прогоне: {', '.join(missing)}" if missing else ""
        return runs[runs["config"].isin(present)], note
    return runs, (
        f"PRIMARY_COMPARISON представлен в прогоне {len(present)} конфигурациями из "
        f"{len(PRIMARY_COMPARISON)}, тест считается по ВСЕМ конфигурациям сетки"
    )


def _block_pivot(runs: pd.DataFrame, metric: str, *, by: str = "block") -> pd.DataFrame:
    pivot = runs.pivot_table(index=by, columns="config", values=metric, aggfunc="mean")
    return pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")


def _budget_text(runs: pd.DataFrame, alarm_budget_per_hour: float | None) -> str:
    """Бюджет тревог прогона для текста проверок: явный аргумент, иначе из ``runs``."""
    if alarm_budget_per_hour is not None:
        return f"{alarm_budget_per_hour:g}"
    if "alarm_budget_per_hour" in runs.columns:
        vals = sorted(runs["alarm_budget_per_hour"].dropna().unique())
        if vals:
            return ", ".join(f"{v:g}" for v in vals)
    return "?"


def validate_protocol(
    runs: pd.DataFrame, prevalence: float, *, alarm_budget_per_hour: float | None = None
) -> list[Check]:
    """Проверить, что метрики не сломаны, **до** интерпретации результатов.

    Пункты воспроизводят предсказания литературы на наших данных:

    * случайный score должен давать ``average_precision ≈ prevalence``
      и ``padf ≈ 0`` — иначе основная метрика обманываема;
    * случайный score должен получать **высокий** PA-F1 — если нет,
      значит реализация PA или структура событий нетипичны, и зазор
      «PA против event-level» нельзя предъявлять как аргумент;
    * обученная референсная модель должна превосходить свою же
      необученную версию — иначе архитектурные различия обсуждать
      бессмысленно (Kim et al., 2021).

    ``alarm_budget_per_hour`` нужен только для текста деталей; по умолчанию
    бюджет берётся из колонки ``runs`` (значение, при котором прогон шёл).
    """
    checks: list[Check] = []
    agg = runs.groupby("config").agg(
        ap=("average_precision", "mean"),
        padf=("padf", "mean"),
        pa_f1=("pa_f1_INVALID_for_ranking", "mean"),
        recall=("event_recall", "mean"),
    )

    if "ctrl_random" in agg.index:
        r = agg.loc["ctrl_random"]
        ratio = r["ap"] / prevalence if prevalence > 0 else np.nan
        checks.append(
            Check(
                "Случайный score: AP на уровне доли аномалий",
                bool(np.isfinite(ratio) and ratio < 1.6),
                f"AP={r['ap']:.5f}, prevalence={prevalence:.5f}, отношение={ratio:.2f} "
                f"(ожидается < 1.6)",
            )
        )
        checks.append(
            Check(
                "Случайный score: padf у пола",
                bool(r["padf"] < 0.25),
                f"padf={r['padf']:.4f} (ожидается < 0.25 при бюджете "
                f"{_budget_text(runs, alarm_budget_per_hour)} тревог/ч)",
            )
        )
        checks.append(
            Check(
                "Случайный score: PA-F1 завышен (воспроизведение Kim et al.)",
                bool(r["pa_f1"] > 0.3),
                f"PA-F1={r['pa_f1']:.3f} при честном recall={r['recall']:.3f}. "
                "Высокое значение здесь — ожидаемый результат, подтверждающий непригодность PA.",
                critical=False,
            )
        )

    if REFERENCE in agg.index and "ctrl_untrained" in agg.index:
        trained = agg.loc[REFERENCE, "padf"]
        untrained = agg.loc["ctrl_untrained", "padf"]
        gain = trained - untrained
        checks.append(
            Check(
                "Обучение даёт прирост над необученной моделью",
                bool(gain > 0.02),
                f"padf обученной={trained:.3f}, необученной={untrained:.3f}, "
                f"прирост={gain:+.3f} (ожидается > 0.02)",
            )
        )

    # --- случайный контроль не должен обходить обученные модели ---
    # Порог «padf у пола» необходим, но недостаточен: контроль может быть
    # ниже абсолютного порога и при этом выше половины таблицы. Именно это
    # произошло на первом CV-прогоне FT-AED, и поймала это только проверка
    # относительного положения.
    if "ctrl_random" in agg.index:
        trained = agg.drop(index=[i for i in ("ctrl_random", "ctrl_untrained") if i in agg.index])
        if not trained.empty:
            beaten = trained[trained["padf"] < agg.loc["ctrl_random", "padf"]]
            checks.append(
                Check(
                    "Случайный контроль не обходит обученные модели",
                    len(beaten) <= len(trained) * 0.25,
                    f"случайный padf={agg.loc['ctrl_random', 'padf']:.3f} выше, чем у "
                    f"{len(beaten)} из {len(trained)} остальных конфигураций"
                    + (f": {', '.join(beaten.index[:5])}" if len(beaten) else ""),
                )
            )

    # --- мощность: способен ли дизайн вообще обнаружить различие ---
    # Критическая разница Nemenyi растёт как sqrt(k(k+1)/6N). При малом
    # числе блоков N и большом числе методов k она может превысить весь
    # размах средних рангов — тогда тест не может отвергнуть равенство
    # ни при каких данных, и «различий не обнаружено» означает
    # «эксперимент недостаточно мощный», а не «методы одинаковы».
    # Считается по предзарегистрированному подмножеству PRIMARY_COMPARISON:
    # бейзлайны и контроли в выводной тест не входят.
    sub, sub_note = primary_runs(runs)
    pivot = _block_pivot(sub, "padf")
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        ranks = mean_ranks(pivot)
        spread = float(ranks.max() - ranks.min())
        cd = nemenyi_cd(pivot.shape[1], pivot.shape[0])
        need = int(np.ceil(pivot.shape[1] * (pivot.shape[1] + 1) / 6
                           * (3.268 / max(spread, 1e-9)) ** 2))
        checks.append(
            Check(
                "Мощность: критическая разница меньше размаха рангов",
                spread > cd,
                f"размах средних рангов {spread:.2f}, CD={cd:.2f} при {pivot.shape[0]} блоках "
                f"и {pivot.shape[1]} методах (PRIMARY_COMPARISON). "
                + (
                    "Дизайн способен обнаружить различие."
                    if spread > cd else
                    f"CD превышает весь размах — тест не может отвергнуть равенство НИ ПРИ КАКИХ "
                    f"данных. Нужно около {need} блоков (сид × фолд) при текущем числе методов, "
                    f"либо меньше методов в одном сравнении."
                )
                + (f" Примечание: {sub_note}." if sub_note else ""),
            )
        )

        # Блок «датасет × сид» считает сиды одного фолда независимыми повторами,
        # хотя данные у них общие. Консервативная оценка — усреднить сиды внутри
        # фолда и взять блоком фолд. Информационная проверка: критической не
        # является, но показывает, насколько мощность держится на сидах.
        if "dataset" in sub.columns and sub["seed"].nunique() > 1:
            fold_pivot = _block_pivot(sub, "padf", by="dataset")
            if fold_pivot.shape[0] >= 2 and fold_pivot.shape[1] >= 3:
                f_ranks = mean_ranks(fold_pivot)
                f_spread = float(f_ranks.max() - f_ranks.min())
                f_cd = nemenyi_cd(fold_pivot.shape[1], fold_pivot.shape[0])
                checks.append(
                    Check(
                        "Мощность на уровне фолдов (сиды усреднены внутри фолда)",
                        f_spread > f_cd,
                        f"размах {f_spread:.2f}, CD={f_cd:.2f} при {fold_pivot.shape[0]} фолдах. "
                        "Сиды одного фолда не независимы, поэтому это консервативная оценка; "
                        "расхождение с проверкой выше означает, что мощность держится на сидах.",
                        critical=False,
                    )
                )

    if "budget_within_tolerance" in runs.columns:
        bad = runs.loc[runs["budget_within_tolerance"] == False, "config"].unique()  # noqa: E712
        checks.append(
            Check(
                "Бюджет параметров выровнен во всех обучаемых конфигурациях",
                len(bad) == 0,
                "все в допуске" if len(bad) == 0 else f"вне допуска: {', '.join(map(str, bad))}",
                critical=False,
            )
        )
    return checks


def rank_table(runs: pd.DataFrame, *, metric: str = "padf", higher_is_better: bool = True) -> pd.DataFrame:
    """Сводная таблица: среднее ± стд, ранг, и всё операционно важное.

    ``n_timed`` — число клеток с замеренным временем вывода: у клеток,
    возобновлённых из старых чекпойнтов, его нет, и среднее ``inference_ms``
    считается по меньшему числу клеток, чем остальные колонки.
    """
    agg = runs.groupby(["config", "label", "group_label"], as_index=False).agg(
        padf=("padf", "mean"),
        padf_std=("padf", "std"),
        event_recall=("event_recall", "mean"),
        event_recall_std=("event_recall", "std"),
        median_delay_min=("median_delay_min", "mean"),
        average_precision=("average_precision", "mean"),
        ap_lift=("ap_lift_over_random", "mean"),
        alarms_per_hour=("alarms_per_hour", "mean"),
        fpr_observed=("fpr_observed", "mean"),
        pa_f1=("pa_f1_INVALID_for_ranking", "mean"),
        pa_inflation=("pa_inflation_ratio", "mean"),
        n_params=("n_params", "mean"),
        inference_ms=("inference_ms_per_window", "mean"),
        n_timed=("inference_ms_per_window", "count"),
        n_runs=("padf", "size"),
    )
    # ранги — только внутри предзарегистрированного подмножества; остальные
    # конфигурации в таблице описательно (ранг «—»)
    sub, _ = primary_runs(runs)
    pivot = _block_pivot(sub, metric)
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
        agg["mean_rank"] = agg["config"].map(ranks)
    else:
        agg["mean_rank"] = np.nan
    agg["in_primary"] = agg["mean_rank"].notna()
    return agg.sort_values(["in_primary", "mean_rank", metric],
                           ascending=[False, True, not higher_is_better]).reset_index(drop=True)


def compare_to_reference(
    runs: pd.DataFrame, *, metric: str = "padf", reference: str = REFERENCE
) -> pd.DataFrame:
    """Парный bootstrap каждой конфигурации против референсного ST-GNN.

    Диаграмма CD даёт общую картину, но главный вопрос диссертации —
    попарный: **обходит ли кандидат референс значимо**. Тест парный,
    потому что блоки (датасет × сид) одни и те же.
    """
    pivot = runs.pivot_table(index="block", columns="config", values=metric, aggfunc="mean")
    if reference not in pivot.columns:
        return pd.DataFrame()
    base = pivot[reference].to_numpy(dtype=float)
    rows = []
    for cfg in pivot.columns:
        if cfg == reference:
            continue
        res = paired_bootstrap_test(pivot[cfg].to_numpy(dtype=float), base)
        label = runs.loc[runs["config"] == cfg, "label"].iloc[0]
        rows.append({
            "config": cfg, "label": label, **res,
            "verdict": (
                "лучше значимо" if res["p_value"] < 0.05 and res["mean_diff"] > 0
                else "хуже значимо" if res["p_value"] < 0.05 and res["mean_diff"] < 0
                else "неразличимо"
            ),
        })
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def axis_importance(runs: pd.DataFrame, *, metric: str = "padf") -> pd.DataFrame:
    """Что объясняет больше разброса: энкодер или механизм score."""
    sub = runs[runs["encoder"].notna() & runs["head"].notna()].copy()
    if sub["encoder"].nunique() < 2 or sub["head"].nunique() < 2:
        return pd.DataFrame()
    return variance_decomposition(sub, value=metric)


def _cross_cover(sub: pd.DataFrame) -> tuple[int, int, int]:
    """``(уровней энкодера, уровней головы, присутствующих пар)`` в подмножестве."""
    if sub.empty:
        return 0, 0, 0
    pairs = sub[["encoder", "head"]].drop_duplicates()
    return int(sub["encoder"].nunique()), int(sub["head"].nunique()), len(pairs)


def _deep_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Прогоны с энкодером и головой, без протокольных контролей (``ctrl_untrained``)."""
    if "encoder" not in runs.columns or "head" not in runs.columns:
        return runs.iloc[0:0]
    mask = runs["encoder"].notna() & runs["head"].notna()
    if "group" in runs.columns:
        mask &= runs["group"] != "control"
    return runs[mask]


def _complete_blocks(sub: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    """Только блоки, в которых есть все ``n_cells`` пар энкодер × голова.

    Пара, упавшая в части блоков (``failures.csv``), делает дизайн
    несбалансированным, и eta² снова смешивает оси с составом блоков.
    """
    if "block" not in sub.columns:
        return sub
    per_block = sub[["block", "encoder", "head"]].drop_duplicates().groupby("block").size()
    return sub[sub["block"].isin(per_block[per_block == n_cells].index)]


def _cross_candidates(runs: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Подмножества, проверяемые на полный крест: все обучаемые, затем клетки ``extended``."""
    deep = _deep_runs(runs)
    ext = deep[deep["config"].isin({c.name for c in EXTENDED})]
    return [("обучаемых конфигураций", deep)] + ([("сетки extended", ext)] if not ext.empty else [])


def full_cross(runs: pd.DataFrame) -> pd.DataFrame:
    """Подмножество прогонов, образующее полный крест энкодер × голова, или пустая таблица.

    R3 сравнивает eta² двух осей, и это сравнение осмысленно только если
    каждая пара уровней присутствует: на нескрещённой сетке (``core``: 4
    энкодера × 5 голов на 6 клетках) оси смешаны, и eta² одной оси
    наполовину объясняется другой. Сначала проверяются все обучаемые
    конфигурации без контролей, затем — только клетки сетки ``extended``.
    Нужно не меньше двух уровней на каждой оси; берутся только блоки, где
    присутствуют все пары (:func:`_complete_blocks`).
    """
    for _, sub in _cross_candidates(runs):
        n_enc, n_head, n_pairs = _cross_cover(sub)
        if n_enc >= 2 and n_head >= 2 and n_pairs == n_enc * n_head:
            sub = _complete_blocks(sub, n_pairs)
            if not sub.empty:
                return sub
    return runs.iloc[0:0]


def _cross_gap(runs: pd.DataFrame) -> str | None:
    """Почему полного креста нет — текст для R3 и раздела 4.

    ``None``, если крест есть или пар энкодер × голова меньше двух
    (сравнивать оси не на чем).
    """
    if not full_cross(runs).empty:
        return None
    cands = _cross_candidates(runs)
    if _cross_cover(cands[0][1])[2] < 2:
        return None
    # знаменатель — у подмножества, ближайшего к кресту: при наличии клеток
    # extended это они, а не весь грид с нескрещёнными головами ядра
    where, sub = cands[-1]
    n_enc, n_head, n_pairs = _cross_cover(sub)
    if n_enc < 2 or n_head < 2:
        axis = "энкодера" if n_enc < 2 else "головы"
        return (f"у оси {axis} среди {where} один уровень ({n_enc}×{n_head}) — "
                f"разложение дисперсии по осям не определено")
    if n_pairs == n_enc * n_head:
        return (f"все {n_pairs} пар {n_enc}×{n_head} {where} присутствуют, но ни в одном блоке "
                f"нет их всех сразу (часть клеток упала) — дизайн несбалансирован")
    return (f"сетка не является полным крестом: присутствует {n_pairs} из {n_enc}×{n_head} = "
            f"{n_enc * n_head} пар энкодер × голова {where}, оси смешаны")


# --------------------------------------------------------------------- правила
#: Действие правила, когда R0 провален: числа остаются, вывода нет.
BLOCKED_R0 = "не оценивается: R0 провален"
#: Действие R3 без полного креста энкодер × голова.
NEEDS_EXTENDED = "не оценивается: требует сетки extended"
#: Префикс действия R2–R7, заблокированных исходом R1 (DECISION_RULES, приоритет 2).
BLOCKED_R1 = "заблокировано R1"
_R1_BLOCKED_RULES = ("R2", "R3", "R4", "R5", "R6", "R7")


def _rule_id(entry: dict[str, str]) -> str:
    return entry["rule"].split(" ", 1)[0]


def decide(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    reference: str = REFERENCE,
    checks: list[Check] | None = None,
) -> list[dict[str, str]]:
    """Применить решающие правила из docs/DECISION_RULES.md.

    Каждое правило: условие на числах → однозначная рекомендация для
    конкретной главы диссертации. Правила зафиксированы заранее.

    Блокировки из раздела «Приоритет при конфликте правил» применяются
    здесь же, а не оставляются читателю отчёта:

    * провал любой критической проверки ``validate_protocol`` (R0) — у всех
      правил остаётся наблюдение, действие заменяется на «не оценивается»;
    * R1 «отрыв НЕ значим» или R1 не оценён — R2–R7 заблокированы;
    * R3 считается только на полном кресте энкодер × голова (:func:`full_cross`).

    ``checks`` — результат ``validate_protocol``; если не передан, считается
    здесь, чтобы гейт R0 нельзя было забыть. У каждой записи есть ``status``:
    ``ok``, ``blocked_r0``, ``blocked_r1``, ``needs_extended`` или ``failed`` (сам R0).
    """
    if checks is None:
        # та же величина, что в manifest: доля аномалий на датасет, а не на клетку
        prevalence = (float(runs.groupby("dataset")["prevalence"].first().mean())
                      if "prevalence" in runs.columns else float("nan"))
        checks = validate_protocol(runs, prevalence)
    failed = [c for c in checks if c.critical and not c.passed]

    out: list[dict[str, str]] = []
    agg = runs.groupby("config")[metric].mean()
    cmp = compare_to_reference(runs, metric=metric, reference=reference)
    verdict = dict(zip(cmp.get("config", []), cmp.get("verdict", []))) if not cmp.empty else {}

    # мощность гейтит все правила: при CD больше размаха «различий нет»
    # означает «эксперимент не мог их увидеть», а не «методы равны»
    pivot = _block_pivot(primary_runs(runs)[0], metric)
    underpowered = False
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        rk = mean_ranks(pivot)
        underpowered = float(rk.max() - rk.min()) <= nemenyi_cd(pivot.shape[1], pivot.shape[0])
    if underpowered:
        out.append({
            "rule": "R0 — мощность эксперимента",
            "finding": (
                f"Блоков {pivot.shape[0]}, методов {pivot.shape[1]}: критическая разница "
                f"превышает размах средних рангов."
            ),
            "action": (
                "ВСЕ ВЫВОДЫ НИЖЕ УСЛОВНЫ. При таком числе блоков тест не способен отвергнуть "
                "равенство методов ни при каких данных, поэтому «неразличимо» здесь означает "
                "«эксперимент не мог увидеть различие», а не «методы одинаковы». "
                "Прежде чем переносить что-либо в диссертацию, увеличить число сидов."
            ),
        })

    # ---- R1: превосходит ли глубина линейную границу
    r1_block: str | None = "R1 в этом прогоне не оценён (нет base_pca или референса)"
    if "base_pca" in agg.index and reference in agg.index:
        pca_cmp = cmp[cmp["config"] == "base_pca"] if not cmp.empty else pd.DataFrame()
        beats_pca = (not pca_cmp.empty) and pca_cmp.iloc[0]["verdict"] == "хуже значимо"
        r1_block = None if beats_pca else (
            "отрыв референса от PCA не значим, глубина не окупается — выбор между глубокими "
            "архитектурами не является предметом работы (DECISION_RULES, приоритет 2)"
        )
        out.append({
            "rule": "R1 — оправдана ли глубокая модель",
            "finding": (
                f"Референс ({reference}) против PCA: "
                f"{'отрыв значим' if beats_pca else 'отрыв НЕ значим'} "
                f"(padf {agg[reference]:.3f} против {agg['base_pca']:.3f})."
            ),
            "action": (
                "Глубокая архитектура обоснована; можно строить главу 2 на ней."
                if beats_pca else
                "ГЛАВНЫЙ РЕЗУЛЬТАТ ДЛЯ ГЛАВЫ 6: линейный бейзлайн не отличим от ST-GNN. "
                "Это публикуемый негативный результат (ср. Sehili 2023, Alves 2026), и он "
                "обязывает пересмотреть постановку, а не менять архитектуру."
            ),
        })

    # ---- R2: окупается ли пространственный prior
    sub = runs[runs["encoder"].notna()]
    if "uses_graph" in sub.columns and sub["uses_graph"].nunique() == 2:
        g = sub.groupby("uses_graph")[metric].mean()
        delta = float(g.get(True, np.nan) - g.get(False, np.nan))
        out.append({
            "rule": "R2 — нужен ли граф",
            "finding": f"Графовые энкодеры против неграфовых: Δ{metric} = {delta:+.3f}.",
            "action": (
                "Пространственный prior окупается — подтверждает вывод бенчмарка FT-AED; "
                "графовый энкодер остаётся основой главы 2."
                if delta > 0.01 else
                "Прирост от графа мал или отрицателен. Это противоречит бенчмарку FT-AED и "
                "требует отдельного разбора: проверьте построение смежности и долю "
                "пропусков по узлам (диагностика в духе Dong et al., 2026) прежде, "
                "чем выносить вывод в диссертацию."
            ),
        })

    # ---- R3: какая ось важнее — только на полном кресте энкодер × голова
    cross = full_cross(runs)
    eta = variance_decomposition(cross, value=metric) if not cross.empty else pd.DataFrame()
    gap = _cross_gap(runs)
    if gap is not None:
        out.append({
            "rule": "R3 — энкодер или механизм score",
            "finding": gap[0].upper() + gap[1:] + ".",
            # одна extended не содержит base_pca и референса: R1 там не оценим, и R3
            # всё равно заблокирован — оценить его может только сетка all
            "action": f"{NEEDS_EXTENDED} (в составе `all` = core + extended)",
            "status": "needs_extended",
        })
    if not eta.empty and len(eta) == 2:
        e = dict(zip(eta["axis"], eta["eta2"]))
        enc, head = e.get("encoder", np.nan), e.get("head", np.nan)
        if np.isfinite(enc) and np.isfinite(head):
            out.append({
                "rule": "R3 — энкодер или механизм score",
                "finding": (
                    f"eta² энкодера = {enc:.3f}, eta² головы = {head:.3f} "
                    f"(полный крест: {cross['config'].nunique()} конфигураций)."
                ),
                "action": (
                    "Механизм score объясняет больше разброса — ГЛАВНЫЙ АРГУМЕНТ диссертации: "
                    "вклад следует делать в механизм (критик / плотность / физика), "
                    "а не в очередную графовую свёртку."
                    if head > enc * 1.2 else
                    "Энкодер объясняет больше разброса — вклад логичнее делать в "
                    "пространственно-временную архитектуру; тогда выбор головы фиксируется "
                    "реконструкцией как референсной."
                    if enc > head * 1.2 else
                    "Оси сопоставимы по вкладу: нужен совместный выбор пары, и в диссертации "
                    "это стоит заявить прямо — одна ось не доминирует."
                ),
            })

    # ---- R4: что брать для контура устранения (калиброванная уверенность)
    if "cand_hypergraph_flow" in agg.index:
        v = verdict.get("cand_hypergraph_flow", "неразличимо")
        out.append({
            "rule": "R4 — голова для safety-валидатора",
            "finding": f"Плотностная голова (гиперграф + flow) против референса: {v}.",
            "action": (
                "Брать её как детектор для контура устранения (глава 5): из всех вариантов "
                "только -log p(h) даёт калиброванный вероятностный score, нужный "
                "safety-валидатору, и при этом она не проигрывает референсу."
                if v in {"лучше значимо", "неразличимо"} else
                "Плотностная голова проигрывает по детекции; для главы 5 придётся "
                "калибровать score референсной модели отдельно (conformal p-value), "
                "а не брать плотность напрямую."
            ),
        })

    # ---- R5: несёт ли критик информацию (пробел D*=1/2)
    if "cand_gcngru_bigan" in agg.index and reference in agg.index:
        v = verdict.get("cand_gcngru_bigan", "неразличимо")
        out.append({
            "rule": "R5 — критик GAN как anomaly score",
            "finding": (
                f"BiGAN-критик при том же энкодере против реконструкции: {v} "
                f"(padf {agg['cand_gcngru_bigan']:.3f} против {agg[reference]:.3f})."
            ),
            "action": (
                "Состязательный механизм даёт прирост — подтверждает линию CCB-GraphGAN; "
                "далее обязательна абляция cycle против critic, чтобы показать, что вклад "
                "даёт именно критик, а не цикл-реконструкция."
                if v == "лучше значимо" else
                "Критик не даёт прироста. Это эмпирическое подтверждение теоретического "
                "пробела D*=1/2: выход критика не несёт информации об аномалии. "
                "Вывод для главы 2: не строить детектор на критике; "
                "сам этот результат оформить как отдельный вклад."
            ),
        })

    # ---- R6: physics-informed как generator-independent детектор
    if "cand_gcngru_physics" in agg.index:
        v = verdict.get("cand_gcngru_physics", "неразличимо")
        share = runs.loc[runs["config"] == "cand_gcngru_physics", "physics_correction_share"]
        share_txt = f"; доля обученной поправки в score = {share.mean():.2f}" if share.notna().any() else ""
        out.append({
            "rule": "R6 — physics-informed голова",
            "finding": f"Невязка LWR против референса: {v}{share_txt}.",
            "action": (
                "Сильный кандидат: единственный механизм, не зависящий от обученного "
                "генератора, то есть структурно невосприимчивый к циркулярности оценки. "
                "Выносить в отдельный раздел главы 2 как незанятую нишу."
                if v in {"лучше значимо", "неразличимо"} else
                "Как самостоятельный детектор слаба, но её невязку стоит оставить как "
                "дополнительный признак или как регуляризатор генератора в главе 4 — "
                "и честно указать в тексте, что как голова детектора она не выиграла."
            ),
        })

    # ---- R7: классический AID
    for aid in ("base_california", "base_snd"):
        if aid in agg.index and reference in agg.index and agg[aid] >= agg[reference] * 0.9:
            out.append({
                "rule": "R7 — классический AID неожиданно конкурентен",
                "finding": f"{aid}: padf {agg[aid]:.3f} против {agg[reference]:.3f} у референса.",
                "action": (
                    "Обязательно обсудить в диссертации: доменный алгоритм 1970-х почти "
                    "догоняет ST-GNN. Либо постановка слишком проста, либо глубокие модели "
                    "не используют физику, доступную классике. И то и другое — содержательный вывод."
                ),
            })
            break

    for r in out:
        r.setdefault("status", "ok")

    # ---- блокировки (DECISION_RULES, «Приоритет при конфликте правил»)
    if failed:
        # словесный вердикт в наблюдении («отрыв НЕ значим», «хуже значимо») —
        # тоже вывод; при R0 он остаётся только справочно
        for r in out:
            r["action"], r["status"] = BLOCKED_R0, "blocked_r0"
            r["finding"] = "Справочно, без вывода: " + r["finding"]
        out.insert(0, {
            "rule": "R0 — валидация протокола",
            "finding": "Критические проверки провалены: " + "; ".join(c.name for c in failed) + ".",
            "action": (
                "СТОП. Выводы по правилам не делаются, пока протокол не исправлен; "
                "числа в наблюдениях ниже приведены справочно."
            ),
            "status": "failed",
        })
    elif r1_block is not None:
        for r in out:
            if _rule_id(r) in _R1_BLOCKED_RULES:
                r["action"], r["status"] = f"{BLOCKED_R1}: {r1_block}.", "blocked_r1"
    return out


RULE_STATUS: dict[str, str] = {
    "failed": "протокол не прошёл валидацию",
    "blocked_r0": "заблокировано: R0 провален",
    "blocked_r1": "заблокировано исходом R1",
    "needs_extended": "вне области определения правила: нужен полный крест",
}


def _ms_cell(r: pd.Series) -> str:
    """Ячейка «мс/окно»: «*» — среднее не по всем клеткам, «—» — замеров нет."""
    if r["n_timed"] == 0 or not np.isfinite(r["inference_ms"]):
        return "—"
    return f"{r['inference_ms']:.2f}" + ("*" if r["n_timed"] < r["n_runs"] else "")


def write_results_md(
    runs: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    out_path: str | Path,
    manifest: dict,
    figures: list[str],
    metric: str = "padf",
    source_note: str | None = None,
) -> Path:
    """Собрать ``reports/RESULTS.md`` — итоговый документ эксперимента.

    ``source_note`` заменяет первую фразу «Сгенерировано автоматически…» —
    для отчётов, пересобранных из сохранённых таблиц другим коммитом кода.
    """
    out_path = Path(out_path)
    prevalence = float(np.mean(list(manifest.get("prevalence", {1: 0.01}).values())))
    checks = validate_protocol(runs, prevalence)
    table = rank_table(runs, metric=metric)
    cmp = compare_to_reference(runs, metric=metric)
    eta = axis_importance(runs, metric=metric)
    rules = decide(runs, metric=metric, checks=checks)

    sub, sub_note = primary_runs(runs)
    pivot = _block_pivot(sub, metric)
    fried = friedman(pivot) if pivot.shape[1] >= 3 else {"p_value": float("nan")}
    cd = nemenyi_cd(pivot.shape[1], pivot.shape[0]) if pivot.shape[1] >= 3 else float("nan")

    L: list[str] = []
    L.append("# Результаты: сравнение архитектур детекции аномалий трафика\n")
    env = manifest.get("environment", {})
    L.append(
        (source_note.rstrip(".") + ". " if source_note else
         "Сгенерировано автоматически из `reports/runs.csv`. ")
        + f"Коммит `{env.get('git_sha', '?')}`, torch {env.get('torch', '?')}, "
        f"устройство {env.get('device_name', '?')}. "
        f"Сидов на конфигурацию: **{manifest.get('n_seeds', '?')}** "
        f"(агрегация — среднее, best-of-N запрещён).\n"
    )

    # ---------------------------------------------------------- 1. валидация
    L.append("## 1. Валидация протокола\n")
    L.append(
        "Выводы об архитектурах имеют смысл только если метрика не обманываема. "
        "Эти проверки воспроизводят предсказания литературы на наших данных.\n"
    )
    L.append("| Проверка | Статус | Детали |")
    L.append("|---|---|---|")
    for c in checks:
        L.append(f"| {c.name} | **{c.mark}** | {c.detail} |")
    critical_failed = [c for c in checks if c.critical and not c.passed]
    if critical_failed:
        L.append(
            "\n> **СТОП.** Критические проверки провалены: "
            + "; ".join(c.name for c in critical_failed)
            + ". Интерпретация остальных разделов недействительна, пока это не исправлено.\n"
        )
    else:
        L.append("\nВсе критические проверки пройдены — таблицу ниже можно читать.\n")

    # ---------------------------------------------------------- 2. таблица
    L.append("## 2. Сводная таблица\n")
    L.append(
        f"Ранжирование по `{metric}` (затухающая награда за раннее обнаружение). "
        f"Рабочая точка — **{manifest.get('alarm_budget_per_hour', 1.0)} подтверждённая ложная тревога в час "
        f"на всю сеть**, подтверждение {manifest.get('persistence', 3)} окна подряд. "
        f"Тест Фридмана: p = {fried.get('p_value', float('nan')):.2e}, "
        f"критическая разница рангов CD = {cd:.2f} — **только по предзарегистрированному "
        f"подмножеству** ({pivot.shape[1]} методов × {pivot.shape[0]} блоков: "
        f"{', '.join(pivot.columns)}). Ранг указан у них; бейзлайны и протокольные "
        f"контроли в сводной таблице описательно, в выводной тест не входят."
        + (f" Примечание: {sub_note}." if sub_note else "")
        + "\n"
    )
    cols = ["label", "group_label", "mean_rank", "padf", "padf_std", "event_recall",
            "median_delay_min", "average_precision", "ap_lift", "alarms_per_hour",
            "fpr_observed", "n_params", "inference_ms"]
    head = ["Конфигурация", "Группа", "Ранг (PRIMARY)", "padf", "±", "event-recall",
            "задержка, мин", "AP", "AP/случайный", "тревог/ч (факт.)", "факт. FPR",
            "параметров", "мс/окно"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for _, r in table.iterrows():
        def f(v, spec=".3f"):
            return "—" if not np.isfinite(v) else format(v, spec)
        L.append(
            f"| {r['label']} | {r['group_label']} | {f(r['mean_rank'], '.2f')} | "
            f"{f(r['padf'])} | {f(r['padf_std'])} | {f(r['event_recall'])} | "
            f"{f(r['median_delay_min'], '+.1f')} | {f(r['average_precision'], '.4f')} | "
            f"{f(r['ap_lift'], '.1f')}× | {f(r['alarms_per_hour'], '.2f')} | "
            f"{f(r['fpr_observed'], '.4f')} | "
            f"{int(r['n_params']) if np.isfinite(r['n_params']) else '—'} | "
            f"{_ms_cell(r)} |"
        )
    # у клеток, возобновлённых без замера времени (старые чекпойнты), времени нет:
    # среднее мс/окно по остальным клеткам помечается, а не выдаётся за полное
    partial = table[(table["n_timed"] > 0) & (table["n_timed"] < table["n_runs"])]
    untimed = table[table["n_timed"] == 0]
    if not partial.empty:
        L.append(
            "\n\\* — среднее по k из n клеток: у остальных замера времени нет (как правило, "
            "возобновлены через --resume из чекпойнта без замера): " + "; ".join(
                f"{r['label']} — {int(r['n_timed'])} из {int(r['n_runs'])}"
                for _, r in partial.iterrows()
            ) + ".\n"
        )
    if not untimed.empty:
        L.append(
            "\n«—» в колонке «мс/окно» — замеров времени нет ни в одной клетке: "
            + ", ".join(str(r["label"]) for _, r in untimed.iterrows()) + ".\n"
        )
    L.append(
        f"\n**Реальная частота ложных тревог.** Порог калибруется по окнам дней валидации на "
        f"номинальный бюджет ({manifest.get('alarm_budget_per_hour', 1.0)}/ч), а на тесте реальная "
        f"частота (колонка «тревог/ч») от него отличается и разная у разных моделей: сдвиг "
        f"распределения между днями и малая калибровочная выборка. Поэтому `padf` сравнивается "
        f"при равном НОМИНАЛЬНОМ, а не равном реальном бюджете; модели с большей реальной "
        f"частотой тревог получают преимущество в recall. Читать `padf` вместе с этой колонкой "
        f"и операционными кривыми (фигура 02).\n"
    )
    L.append(
        "\n`PA-F1` сознательно **не** включён в таблицу ранжирования: он непригоден для "
        "сравнения моделей. Его значения и коэффициент инфляции — на фигуре 05 "
        "и в `tables/fig05_metric_inflation.csv`.\n"
    )

    # ---------------------------------------------------------- 3. против референса
    if not cmp.empty:
        L.append(f"## 3. Попарно против референса (`{REFERENCE}`)\n")
        L.append(
            "Парный bootstrap на одних и тех же блоках. Референс — графовый автоэнкодер, "
            "то есть семейство, названное лучшим в бенчмарке FT-AED.\n"
        )
        if critical_failed:
            L.append(
                "Числа приведены справочно: R0 провален, поэтому вердикты не оцениваются.\n"
            )
        L.append("| Конфигурация | Δ padf | 95% ДИ | p | Вердикт |")
        L.append("|---|---|---|---|---|")
        for _, r in cmp.iterrows():
            v = "не оценивается (R0)" if critical_failed else r["verdict"]
            L.append(
                f"| {r['label']} | {r['mean_diff']:+.4f} | "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.3f} | {v} |"
            )
        L.append("")

    # ---------------------------------------------------------- 4. оси
    if not eta.empty:
        L.append("## 4. Что важнее: энкодер или механизм score\n")
        L.append("| Ось | Уровней | eta² (доля объяснённой дисперсии) |")
        L.append("|---|---|---|")
        for _, r in eta.iterrows():
            name = "энкодер" if r["axis"] == "encoder" else "механизм score (голова)"
            L.append(f"| {name} | {int(r['n_levels'])} | {r['eta2']:.3f} |")
        L.append("")
        gap = _cross_gap(runs)
        cross = full_cross(runs)
        if gap is not None:
            L.append(
                f"> Полного креста энкодер × голова нет: {gap}. Таблица выше описательная, "
                f"правило R3 по ней не оценивается (требует сетки `extended` в составе `all` = "
                f"core + extended).\n"
            )
        elif not cross.empty and len(cross) != int((runs["encoder"].notna() & runs["head"].notna()).sum()):
            # таблица выше — по всем конфигурациям с энкодером и головой (вместе с
            # нескрещёнными и ctrl_untrained); R3 считается только на кресте
            L.append(
                f"Таблица выше описательная: все конфигурации с энкодером и головой, включая "
                f"нескрещённые и `ctrl_untrained`. Правило R3 (раздел 5) оценивается только на "
                f"полном кресте из {cross['config'].nunique()} конфигураций:\n"
            )
            L.append("| Ось (полный крест, R3) | Уровней | eta² |")
            L.append("|---|---|---|")
            for _, r in variance_decomposition(cross, value=metric).iterrows():
                name = "энкодер" if r["axis"] == "encoder" else "механизм score (голова)"
                L.append(f"| {name} | {int(r['n_levels'])} | {r['eta2']:.3f} |")
            L.append("")

    # ---------------------------------------------------------- 5. решения
    L.append("## 5. Вывод для диссертации\n")
    L.append(
        "Решающие правила зафиксированы в `docs/DECISION_RULES.md` **до** прогона — "
        "чтобы вывод нельзя было подогнать под полученные числа.\n"
    )
    if critical_failed:
        L.append(
            "> **СТОП.** R0 провален: ни одно правило ниже не оценивается. Наблюдения "
            "оставлены справочно: словесные вердикты в них («значим», «неразличимо») "
            "выводами не являются; действий нет.\n"
        )
    for r in rules:
        L.append(f"### {r['rule']}\n")
        if r.get("status", "ok") != "ok":
            L.append(f"**Статус.** `{r['status']}` — {RULE_STATUS.get(r['status'], r['status'])}\n")
        L.append(f"**Наблюдение.** {r['finding']}\n")
        L.append(f"**Что делать.** {r['action']}\n")

    # ---------------------------------------------------------- 6. фигуры
    L.append("## 6. Фигуры\n")
    captions = {
        "fig01": "Диаграмма критических различий: какие архитектуры различаются значимо.",
        "fig02": "Операционные кривые: event-recall и задержка против FPR.",
        "fig03": "PR-кривые с линией случайного детектора.",
        "fig04": "Теплокарта энкодер × голова: какая ось важнее.",
        "fig05": "Инфляция point-adjustment: почему PA-F1 из литературы несопоставим.",
        "fig06": "Разброс по сидам: отличима ли разница методов от шума.",
        "fig07": "Парето «качество против стоимости вывода».",
        "fig08": "Разбор одного события: кто сработал раньше.",
        "fig09": "Вклад пространственного prior при выровненном бюджете.",
    }
    for fname in sorted(figures):
        stem = Path(fname).stem
        key = stem.split("_")[0]
        L.append(f"**{captions.get(key, stem)}**\n")
        L.append(f"![{stem}](figures/{Path(fname).name})\n")
        L.append(f"Числа фигуры: `tables/{stem}.csv`\n")
        if key == "fig07" and (table["n_timed"] < table["n_runs"]).any():
            L.append(
                "Время вывода у части конфигураций усреднено не по всем клеткам (пометка «*» "
                "на фигуре, колонки `n_timed` и `n_runs` в CSV); конфигураций без замеров "
                "на фигуре нет.\n"
            )

    # ---------------------------------------------------------- 7. оговорки
    L.append("## 7. Ограничения этого прогона\n")
    datasets = list(manifest.get("datasets", {}).keys())
    synth_only = all("synthetic" in d for d in datasets) if datasets else False
    if synth_only:
        L.append(
            "- **Прогон выполнен только на синтетическом коридоре.** Выводы о ранжировании "
            "архитектур нельзя переносить в диссертацию: синтетика нужна для отладки "
            "пайплайна и для контролируемых экспериментов rarity × amplitude. "
            "Итоговая таблица главы 2 строится на FT-AED с реальными метками.\n"
        )
    L.append(
        f"- Число событий в тесте ограничено, поэтому доверительные интервалы считаются "
        f"bootstrap, а не нормальной аппроксимацией.\n"
        f"- Рабочая точка задана в тревогах в час, а не в процентах FPR: поточечный FPR "
        f"на сети из сотен узлов не является операционной величиной и вырождается "
        f"(при FPR 1% случайный score обнаруживает почти любое событие). Фактический FPR "
        f"приведён в таблице для сопоставимости с литературой.\n"
        f"- Задержка измеряется относительно времени **официального отчёта**, которое само "
        f"зашумлено; поэтому отрицательная задержка интерпретируется как «раньше отчёта», "
        f"а не как «раньше события».\n"
        f"- Сравнение честно только при выровненном бюджете параметров: фактические значения — "
        f"в `reports/budget.csv`.\n"
    )
    if manifest.get("n_failures"):
        L.append(f"- Прогонов с ошибкой: {manifest['n_failures']}, подробности в `reports/failures.csv`.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return out_path


# ------------------------------------------------- решающие правила аугментации
def decide_augmentation(summary: pd.DataFrame, *, metric: str = "padf") -> list[dict[str, str]]:
    """Применить правила R8–R10 из docs/DECISION_RULES.md к эффекту синтетики.

    Вход — результат :func:`stad.augment.summarise`: парный эффект
    относительно обучения без синтетики, по энкодерам и источникам.
    Правила зафиксированы до прогона, здесь только подстановка чисел.
    """
    out: list[dict[str, str]] = []
    if summary.empty:
        return [{
            "rule": "R8 — помогает ли синтетика",
            "finding": "Таблица эффекта пуста: арм аугментации не прогонялся.",
            "action": "Запустить scripts/run_augmentation.py прежде, чем делать выводы.",
        }]

    # ---- R8: есть ли эффект вообще
    best = summary.loc[summary["delta_mean"].idxmax()]
    sd = float(best["delta_std"]) if np.isfinite(best["delta_std"]) else 0.0
    significant = best["delta_mean"] > sd and best["delta_mean"] > 0
    out.append({
        "rule": "R8 — помогает ли синтетика вообще",
        "finding": (
            f"Лучший эффект: {best['encoder']} + {best['source']} при ratio={best['ratio']}, "
            f"Δ{metric} = {best['delta_mean']:+.4f} ± {sd:.4f} (n={int(best['n'])})."
        ),
        "action": (
            "Аугментация работает — переходить к R9 и выяснять, за счёт чего."
            if significant else
            "Эффект не отличим от разброса. ПУБЛИКУЕМЫЙ НЕГАТИВНЫЙ РЕЗУЛЬТАТ для главы 4: "
            "он объясняет противоречие в литературе (+13% F1 против нуля) — выигрыш зависит "
            "не от качества генератора, а от того, ограничена ли задача размером редкого класса."
        ),
    })
    if not significant:
        return out

    # ---- R9: GAN или сам факт расширения
    by_src = summary.groupby("source", as_index=False)["delta_mean"].mean()
    d = dict(zip(by_src["source"], by_src["delta_mean"]))
    if "gan" in d and "lwr" in d:
        gap = d["gan"] - d["lwr"]
        pooled = float(summary["delta_std"].mean() or 0.0)
        if abs(gap) <= pooled:
            action = (
                "Помогает РАСШИРЕНИЕ РЕДКОГО КЛАССА КАК ТАКОВОЕ, а не выученное "
                "распределение. Вклад работы — протокол аугментации и его проверка, "
                "а не GAN. Качество генератора доказывать не требуется."
            )
        elif gap > 0:
            action = (
                "GAN даёт дополнительный выигрыш сверх физической инъекции. "
                "УТВЕРЖДЕНИЕ НЕ ЗАЩИЩАЕТСЯ без второй проверки: нужна кросс-валидация "
                "на генераторе другого семейства, иначе нельзя исключить, что детектор "
                "выучил артефакты именно этого WGAN-GP."
            )
        else:
            action = (
                "Физическая инъекция сильнее обученной. Для диссертации это выгодно: "
                "детектор без обученного генератора проще, воспроизводимее и структурно "
                "невосприимчив к циркулярности оценки."
            )
        out.append({
            "rule": "R9 — за счёт чего помогает",
            "finding": f"Δ{metric}: GAN {d['gan']:+.4f}, физика LWR {d['lwr']:+.4f}, "
                       f"разница {gap:+.4f} при типичном разбросе {pooled:.4f}.",
            "action": action,
        })

    # ---- R10: зависит ли от архитектуры
    by_enc = summary.groupby("encoder", as_index=False)["delta_mean"].mean()
    spread = float(by_enc["delta_mean"].max() - by_enc["delta_mean"].min())
    pooled = float(summary["delta_std"].mean() or 0.0)
    out.append({
        "rule": "R10 — зависит ли эффект от архитектуры",
        "finding": "; ".join(f"{r.encoder}: {r.delta_mean:+.4f}" for r in by_enc.itertuples())
                   + f" (разброс {spread:.4f}, типичная стд {pooled:.4f})",
        "action": (
            "Эффект зависит от архитектуры: «какая лучше на чистых данных» и «какая "
            "выигрывает от аугментации» — разные вопросы, и в главе 2 архитектура "
            "выбирается под тот режим, в котором она будет работать."
            if spread > pooled else
            "Эффект ортогонален выбору энкодера: аугментацию можно обсуждать в главе 4 "
            "независимо от главы 2."
        ),
    })
    return out
