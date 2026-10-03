"""Статистическая строгость: Friedman, post-hoc Nemenyi, bootstrap CI.

Без этого блока таблица сравнения — набор чисел без вывода. Три
требования, которые он реализует:

1. **Разницу надо проверять, а не видеть.** Между 12 конфигурациями на
   нескольких датасетах/фолдах сравнение делается тестом Фридмана по
   рангам, затем post-hoc Nemenyi с критической разницей (CD). Это
   стандарт сравнения множества методов на множестве наборов данных.

2. **Одно число ничего не значит.** Alves et al. (2026) показали на 100
   прогонах, что дисперсия между машинами и инициализациями
   сопоставима с разницей между методами. Поэтому каждая конфигурация
   прогоняется на нескольких сидах, и в отчёт идут среднее ± стд.

3. **Никакого best-of-N.** Lyu (2026): при отчётности «лучшее из N»
   часть метрик становится обманываемой. Здесь агрегация —
   **среднее по сидам**, выбор лучшего сида запрещён; число сидов
   фиксируется в конфиге и попадает в отчёт.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Критические значения статистики Nemenyi q_alpha для alpha=0.05
#: по числу сравниваемых методов k (двусторонний тест).
_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}
_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
    15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319,
}


def rank_matrix(pivot: pd.DataFrame, *, higher_is_better: bool = True) -> pd.DataFrame:
    """Ранги методов внутри каждого блока (строки = блоки, столбцы = методы).

    Блок — это один «датасет-сид-фолд»: единица, внутри которой методы
    сравниваются напрямую.
    """
    vals = pivot if higher_is_better else -pivot
    return vals.rank(axis=1, ascending=False)


def friedman(pivot: pd.DataFrame, *, higher_is_better: bool = True) -> dict[str, float]:
    """Тест Фридмана: есть ли вообще различие между методами."""
    clean = pivot.dropna(axis=0, how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_blocks": int(clean.shape[0]),
                "n_methods": int(clean.shape[1]), "note": "недостаточно блоков или методов"}
    cols = [clean[c].to_numpy() for c in clean.columns]
    stat, p = stats.friedmanchisquare(*cols)
    ranks = rank_matrix(clean, higher_is_better=higher_is_better).mean(axis=0)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": int(clean.shape[0]),
        "n_methods": int(clean.shape[1]),
        "best_method": str(ranks.idxmin()),
        "best_mean_rank": float(ranks.min()),
    }


def nemenyi_cd(n_methods: int, n_blocks: int, *, alpha: float = 0.05) -> float:
    """Критическая разница средних рангов.

    .. math:: CD = q_\\alpha \\sqrt{\\frac{k(k+1)}{6N}}

    Две модели различаются значимо, если разность их средних рангов
    превышает CD. Это и есть содержание диаграммы критических различий.
    """
    table = _Q05 if alpha <= 0.05 else _Q10
    k = int(n_methods)
    if k < 2:
        return float("nan")
    q = table.get(k)
    if q is None:  # экстраполяция за пределы таблицы
        q = table[max(table)] + 0.027 * (k - max(table))
    return float(q * np.sqrt(k * (k + 1) / (6.0 * max(1, n_blocks))))


def mean_ranks(pivot: pd.DataFrame, *, higher_is_better: bool = True) -> pd.Series:
    """Средние ранги методов (меньше = лучше), отсортированные."""
    clean = pivot.dropna(axis=0, how="any")
    return rank_matrix(clean, higher_is_better=higher_is_better).mean(axis=0).sort_values()


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap доверительный интервал среднего: ``(mean, lo, hi)``.

    Нужен там, где событий мало (на FT-AED их десятки): нормальная
    аппроксимация при таком N необоснованна.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_test(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Парный bootstrap: значимо ли a лучше b на одних и тех же блоках.

    Используется для главного попарного вывода отчёта («обходит ли
    кандидат референсный ST-GNN») — там, где диаграмма CD даёт только
    общую картину.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("парный тест требует одинаковых форм")
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size == 0:
        return {"mean_diff": float("nan"), "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    p_two_sided = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "mean_diff": float(d.mean()),
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "p_value": float(min(1.0, p_two_sided)),
        "n": int(d.size),
    }


def variance_decomposition(df: pd.DataFrame, *, value: str) -> pd.DataFrame:
    """Какая ось объясняет больше разброса: энкодер или голова.

    Это прямая проверка главной гипотезы бенчмарка. Считается доля
    дисперсии метрики, объяснённая группировкой по ``encoder`` и по
    ``head`` (однофакторный eta-квадрат по каждой оси отдельно), плюс
    остаточная доля, относимая на сид.

    Интерпретация: если ``eta2`` головы заметно выше, чем энкодера,
    то выбор механизма score важнее выбора архитектуры энкодера — и
    диссертации следует сосредоточиться на механизме.
    """
    rows = []
    total = df[value].var(ddof=0)
    for axis in ("encoder", "head"):
        if axis not in df.columns:
            continue
        grand = df[value].mean()
        between = sum(
            len(g) * (g[value].mean() - grand) ** 2 for _, g in df.groupby(axis)
        )
        ss_total = ((df[value] - grand) ** 2).sum()
        rows.append(
            {
                "axis": axis,
                "n_levels": int(df[axis].nunique()),
                "eta2": float(between / ss_total) if ss_total > 0 else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["total_variance"] = float(total)
    return out
