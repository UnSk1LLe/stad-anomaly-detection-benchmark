"""Единая визуальная система для всех фигур отчёта.

Решения и их причины:

* **Палитра валидирована, а не подобрана на глаз.** Восемь категориальных
  оттенков проверены на разделимость при дальтонизме (worst adjacent
  CVD ΔE 9.1 при цели ≥8) и на разделимость для нормального зрения
  (worst adjacent ΔE 19.6 при полу 15). Для диаграмм рассеяния, где
  сравниваются все пары, используются только первые **три** слота —
  полная восьмёрка все-пары не проходит.
* **Цвет никогда не единственный носитель смысла.** Группы дополнительно
  различаются маркерами и прямыми подписями; у каждой фигуры есть
  CSV-двойник в ``reports/tables``.
* **Никаких двух осей Y.** Две величины разного масштаба — это две
  панели, а не два масштаба на одной.
* **Сетка и оси рецессивны**, данные доминируют.
* Рядом с каждым PNG сохраняется PDF: в диссертацию идут векторные
  версии.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Категориальные слоты. Порядок фиксирован и никогда не перебирается
#: заново при смене набора серий: цвет следует сущности, не её рангу.
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

#: Маркеры как вторичное кодирование идентичности: цвет никогда не единственный
#: носитель смысла (печать в оттенках серого, дальтонизм, forced-colors).
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Для диаграмм рассеяния и малых множеств (сравниваются все пары).
CATEGORICAL_SAFE3: tuple[str, ...] = CATEGORICAL[:3]

#: Одна последовательная шкала для величин (магнитуда), светлая → тёмная.
SEQUENTIAL = "Blues"

#: Расходящаяся шкала для знаковых величин (две полярности + нейтраль).
DIVERGING = "RdBu_r"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3de"

#: Устойчивое сопоставление «группа конфигураций → цвет и маркер».
GROUP_STYLE: dict[str, dict[str, str]] = {
    "control": {"color": CATEGORICAL[7], "marker": "x", "label": "Протокольный контроль"},
    "baseline": {"color": CATEGORICAL[1], "marker": "s", "label": "Бейзлайн"},
    "non_graph": {"color": CATEGORICAL[3], "marker": "^", "label": "Без графа"},
    "candidate": {"color": CATEGORICAL[0], "marker": "o", "label": "Кандидат"},
}


def apply_theme() -> None:
    """Глобальные параметры matplotlib. Вызывается один раз перед фигурами."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "text.color": INK_PRIMARY,
        }
    )


def save(fig, out_dir: Path, stem: str, *, table: pd.DataFrame | None = None, tables_dir: Path | None = None) -> list[Path]:
    """Сохранить PNG + PDF и, при наличии, CSV-двойник фигуры.

    CSV обязателен там, где цвет серии имеет контраст ниже 3:1 к фону —
    это требование доступности («relief rule»): смысл должен быть
    доступен без различения цвета. Для научного репозитория это ещё и
    требование воспроизводимости: числа фигуры должны быть читаемы
    машиной.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)

    if table is not None:
        td = Path(tables_dir or out_dir.parent / "tables")
        td.mkdir(parents=True, exist_ok=True)
        tp = td / f"{stem}.csv"
        table.to_csv(tp, index=False, encoding="utf-8")
        paths.append(tp)
    return paths


def annotate_source(fig, text: str) -> None:
    """Подпись-источник внизу фигуры: датасет, число сидов, протокол."""
    fig.text(0.0, -0.02, text, fontsize=8, color=INK_MUTED, ha="left", va="top")


def wrap(labels, width: int = 26) -> list[str]:
    """Перенос длинных подписей, чтобы не было коллизий на оси."""
    import textwrap

    return ["\n".join(textwrap.wrap(str(s), width)) for s in labels]
