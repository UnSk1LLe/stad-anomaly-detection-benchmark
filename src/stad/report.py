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
from .registry import REFERENCE


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


def validate_protocol(runs: pd.DataFrame, prevalence: float) -> list[Check]:
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
                f"padf={r['padf']:.4f} (ожидается < 0.25 при бюджете 1 тревога/ч)",
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
    """Сводная таблица: среднее ± стд, ранг, и всё операционно важное."""
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
        n_runs=("padf", "size"),
    )
    pivot = runs.pivot_table(index="block", columns="config", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
        agg["mean_rank"] = agg["config"].map(ranks)
    else:
        agg["mean_rank"] = np.nan
    return agg.sort_values("mean_rank" if agg["mean_rank"].notna().any() else metric,
                           ascending=agg["mean_rank"].notna().any())


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


# --------------------------------------------------------------------- правила
def decide(
    runs: pd.DataFrame,
    *,
    metric: str = "padf",
    reference: str = REFERENCE,
) -> list[dict[str, str]]:
    """Применить решающие правила из docs/DECISION_RULES.md.

    Каждое правило: условие на числах → однозначная рекомендация для
    конкретной главы диссертации. Правила зафиксированы заранее.
    """
    out: list[dict[str, str]] = []
    agg = runs.groupby("config")[metric].mean()
    cmp = compare_to_reference(runs, metric=metric, reference=reference)
    verdict = dict(zip(cmp.get("config", []), cmp.get("verdict", []))) if not cmp.empty else {}

    # ---- R1: превосходит ли глубина линейную границу
    if "base_pca" in agg.index and reference in agg.index:
        pca_cmp = cmp[cmp["config"] == "base_pca"] if not cmp.empty else pd.DataFrame()
        beats_pca = (not pca_cmp.empty) and pca_cmp.iloc[0]["verdict"] == "хуже значимо"
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

    # ---- R3: какая ось важнее
    eta = axis_importance(runs, metric=metric)
    if not eta.empty and len(eta) == 2:
        e = dict(zip(eta["axis"], eta["eta2"]))
        enc, head = e.get("encoder", np.nan), e.get("head", np.nan)
        if np.isfinite(enc) and np.isfinite(head):
            out.append({
                "rule": "R3 — энкодер или механизм score",
                "finding": f"eta² энкодера = {enc:.3f}, eta² головы = {head:.3f}.",
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
    return out


def write_results_md(
    runs: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    out_path: str | Path,
    manifest: dict,
    figures: list[str],
    metric: str = "padf",
) -> Path:
    """Собрать ``reports/RESULTS.md`` — итоговый документ эксперимента."""
    out_path = Path(out_path)
    prevalence = float(np.mean(list(manifest.get("prevalence", {1: 0.01}).values())))
    checks = validate_protocol(runs, prevalence)
    table = rank_table(runs, metric=metric)
    cmp = compare_to_reference(runs, metric=metric)
    eta = axis_importance(runs, metric=metric)
    rules = decide(runs, metric=metric)

    pivot = runs.pivot_table(index="block", columns="config", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="any").dropna(axis=0, how="any")
    fried = friedman(pivot) if pivot.shape[1] >= 3 else {"p_value": float("nan")}
    cd = nemenyi_cd(pivot.shape[1], pivot.shape[0]) if pivot.shape[1] >= 3 else float("nan")

    L: list[str] = []
    L.append("# Результаты: сравнение архитектур детекции аномалий трафика\n")
    env = manifest.get("environment", {})
    L.append(
        f"Сгенерировано автоматически из `reports/runs.csv`. "
        f"Коммит `{env.get('git_sha', '?')}`, torch {env.get('torch', '?')}, "
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
        f"критическая разница рангов CD = {cd:.2f}.\n"
    )
    cols = ["label", "group_label", "mean_rank", "padf", "padf_std", "event_recall",
            "median_delay_min", "average_precision", "ap_lift", "fpr_observed",
            "n_params", "inference_ms"]
    head = ["Конфигурация", "Группа", "Ранг", "padf", "±", "event-recall",
            "задержка, мин", "AP", "AP/случайный", "факт. FPR", "параметров", "мс/окно"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for _, r in table.iterrows():
        def f(v, spec=".3f"):
            return "—" if not np.isfinite(v) else format(v, spec)
        L.append(
            f"| {r['label']} | {r['group_label']} | {f(r['mean_rank'], '.2f')} | "
            f"{f(r['padf'])} | {f(r['padf_std'])} | {f(r['event_recall'])} | "
            f"{f(r['median_delay_min'], '+.1f')} | {f(r['average_precision'], '.4f')} | "
            f"{f(r['ap_lift'], '.1f')}× | {f(r['fpr_observed'], '.4f')} | "
            f"{int(r['n_params']) if np.isfinite(r['n_params']) else '—'} | "
            f"{f(r['inference_ms'], '.2f')} |"
        )
    L.append(
        "\n`PA-F1` сознательно **не** включён в таблицу ранжирования: он непригоден для "
        "сравнения моделей. Его значения и коэффициент инфляции — на фигуре 05 "
        "и в `reports/tables/fig05_metric_inflation.csv`.\n"
    )

    # ---------------------------------------------------------- 3. против референса
    if not cmp.empty:
        L.append(f"## 3. Попарно против референса (`{REFERENCE}`)\n")
        L.append(
            "Парный bootstrap на одних и тех же блоках. Референс — графовый автоэнкодер, "
            "то есть семейство, названное лучшим в бенчмарке FT-AED.\n"
        )
        L.append("| Конфигурация | Δ padf | 95% ДИ | p | Вердикт |")
        L.append("|---|---|---|---|---|")
        for _, r in cmp.iterrows():
            L.append(
                f"| {r['label']} | {r['mean_diff']:+.4f} | "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.3f} | {r['verdict']} |"
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

    # ---------------------------------------------------------- 5. решения
    L.append("## 5. Вывод для диссертации\n")
    L.append(
        "Решающие правила зафиксированы в `docs/DECISION_RULES.md` **до** прогона — "
        "чтобы вывод нельзя было подогнать под полученные числа.\n"
    )
    for r in rules:
        L.append(f"### {r['rule']}\n")
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
        L.append(f"Числа фигуры: `reports/tables/{stem}.csv`\n")

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
