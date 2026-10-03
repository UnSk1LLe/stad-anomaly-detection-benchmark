"""Сквозной эксперимент: данные → сетка → фигуры → отчёт.

Один конфиг управляет всем прогоном, и тот же конфиг позволяет
пересобрать фигуры из сохранённых score без повторного обучения.
Это важно практически: сетка из 12 конфигураций × 5 сидов обучается
часами, а фигуры переделываются десятки раз.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import SplitData, load_dataset
from .metrics import evaluation_view
from .metrics.event_level import threshold_at_alarm_rate
from .registry import REFERENCE, get_grid
from .report import write_results_md
from .runner import run_grid
from .train import TrainConfig
from .viz import (
    apply_theme,
    fig_critical_difference,
    fig_encoder_head_heatmap,
    fig_event_timeline,
    fig_metric_inflation,
    fig_operating_curves,
    fig_pareto,
    fig_pr_curves,
    fig_seed_variance,
    fig_spatial_prior_contribution,
)
from .report import axis_importance


@dataclass
class ExperimentConfig:
    """Полная спецификация прогона. Сохраняется рядом с результатами."""

    name: str = "smoke"
    grid: str = "smoke"
    seeds: tuple[int, ...] = (0, 1, 2)
    param_budget: int = 200_000
    # Рабочая точка: подтверждённых ложных тревог в час на всю сеть.
    # Операционная единица, а не процент: см. threshold_at_alarm_rate.
    alarm_budget_per_hour: float = 1.0
    half_life_min: float = 15.0
    # сколько подряд идущих окон подтверждают тревогу (логика California)
    persistence: int = 3
    primary_metric: str = "padf"
    out_dir: str = "reports"
    datasets: dict[str, dict] = field(default_factory=lambda: {
        "synthetic": {"loader": "synthetic", "kwargs": {}}
    })
    train: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw["seeds"] = tuple(raw.get("seeds", (0, 1, 2)))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def train_config(self) -> TrainConfig:
        known = {f for f in TrainConfig.__dataclass_fields__}
        return TrainConfig(**{k: v for k, v in self.train.items() if k in known})


def build_datasets(cfg: ExperimentConfig) -> dict[str, SplitData]:
    """Собрать все датасеты прогона. Детерминировано по конфигу.

    Поле ``from_file`` в спецификации датасета подключает отдельный YAML
    с параметрами загрузки. Это нужно для FT-AED: сопоставление столбцов
    и радиусы разметки живут в ``configs/data/ft_aed.yaml`` и правятся
    под конкретную версию скачанных файлов, не затрагивая конфиг прогона.
    Значения из ``kwargs`` имеют приоритет над подключённым файлом.
    """
    out: dict[str, SplitData] = {}
    for name, spec in cfg.datasets.items():
        loader = spec.get("loader", name)
        kwargs: dict = {}
        if spec.get("from_file"):
            path = Path(spec["from_file"])
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                raise FileNotFoundError(f"датасет {name}: не найден {path}")
            kwargs.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        kwargs.update(spec.get("kwargs") or {})
        out[name] = load_dataset(loader, **kwargs)
    return out


# --------------------------------------------------------------------- фигуры
def _pr_data(
    runs: pd.DataFrame, data: SplitData, scores_dir: Path, dataset: str, seed: int, top: int = 5
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from sklearn.metrics import precision_recall_curve

    order = (
        runs[runs["dataset"] == dataset]
        .groupby(["config", "label"], as_index=False)["average_precision"].mean()
        .sort_values("average_precision", ascending=False)
    )
    keep = list(order.itertuples())[:top]
    if not any(r.config == "ctrl_random" for r in keep):
        rnd = order[order["config"] == "ctrl_random"]
        if not rnd.empty:
            keep.append(next(rnd.itertuples()))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in keep:
        f = scores_dir / f"{r.config}__{dataset}__seed{seed}.npy"
        if not f.exists():
            continue
        s = np.load(f)
        s, y, _ = evaluation_view(s, data)
        precision, recall, _ = precision_recall_curve(y.ravel(), s.ravel())
        out[r.label] = (recall, precision)
    return out


def _event_traces(
    runs: pd.DataFrame,
    data: SplitData,
    scores_dir: Path,
    dataset: str,
    seed: int,
    *,
    alarm_budget: float,
    step_min: float,
    persistence: int = 3,
    top: int = 4,
    window_min: float = 45.0,
) -> pd.DataFrame:
    """След score вокруг одного события для нескольких лучших моделей.

    Выбирается событие, обнаруженное **не всеми** моделями: именно на
    таких случаях видно различие архитектур, а на тривиальных —
    не видно ничего.
    """
    cand = (
        runs[(runs["dataset"] == dataset) & (runs["group"].isin(["candidate", "non_graph"]))]
        .groupby(["config", "label"], as_index=False)["padf"].mean()
        .sort_values("padf", ascending=False)
        .head(top)
    )
    if cand.empty or data.events.empty:
        return pd.DataFrame()

    loaded: dict[str, tuple[str, np.ndarray, float]] = {}
    for r in cand.itertuples():
        f = scores_dir / f"{r.config}__{dataset}__seed{seed}.npy"
        if f.exists():
            s = np.load(f)
            s, y_ev, _ = evaluation_view(s, data)
            thr = threshold_at_alarm_rate(s, y_ev, alarm_budget, step_min,
                                          persistence=persistence)
            loaded[r.config] = (r.label, s, thr)
    if not loaded:
        return pd.DataFrame()

    # ищем «спорное» событие
    best_eid, best_spread = None, -1
    for ev in data.events.itertuples():
        hits = []
        for _, (_, s, thr) in loaded.items():
            eid_ev = data.event_id_test[:, : s.shape[1]]
            mask = (eid_ev == ev.event_id) & (s > thr)
            hits.append(bool(mask.any()))
        spread = len(hits) - abs(sum(hits) * 2 - len(hits))   # максимум при равном делении
        if spread > best_spread:
            best_eid, best_spread = int(ev.event_id), spread
    if best_eid is None:
        return pd.DataFrame()

    ev = data.events[data.events["event_id"] == best_eid].iloc[0]
    rel = data.t_test - float(ev.t_report)
    sel = np.abs(rel) <= window_min
    nodes = np.where((data.event_id_test == best_eid).any(axis=0))[0]
    if nodes.size == 0:
        nodes = np.arange(data.n_nodes)

    rows = []
    for _, (label, s, thr) in loaded.items():
        # нормировка на общий масштаб: score разных механизмов несопоставимы по
        # абсолютной величине, поэтому приводим к [0,1] по нормальной части теста
        y_ev = data.y_test[:, : s.shape[1]]
        lo = float(np.quantile(s[y_ev == 0], 0.01))
        hi = float(np.quantile(s[y_ev == 0], 0.9999))
        rng = max(hi - lo, 1e-9)
        cols = nodes if s.shape[1] > 1 else np.array([0])
        agg = s[np.ix_(np.where(sel)[0], cols)].max(axis=1)
        rows.append(
            pd.DataFrame({
                "event_id": best_eid,
                "label": label,
                "minutes_from_report": rel[sel],
                "score_norm": (agg - lo) / rng,
                "threshold_norm": (thr - lo) / rng,
            })
        )
    return pd.concat(rows, ignore_index=True)


def make_figures(
    cfg: ExperimentConfig,
    runs: pd.DataFrame,
    curves: pd.DataFrame,
    datasets: dict[str, SplitData],
) -> list[str]:
    """Построить все фигуры отчёта. Возвращает имена файлов PNG."""
    apply_theme()
    out = Path(cfg.out_dir)
    fig_dir, tbl_dir = out / "figures", out / "tables"
    scores_dir = out / "scores"
    metric = cfg.primary_metric
    produced: list[str] = []

    def collect(paths):
        produced.extend(p.name for p in paths if p.suffix == ".png")

    collect(fig_critical_difference(
        runs, metric=metric,
        higher_is_better=metric != "median_delay_min",
        out_dir=fig_dir, tables_dir=tbl_dir,
    ))

    if not curves.empty:
        top = (
            runs.groupby("label", as_index=False)[metric].mean()
            .sort_values(metric, ascending=False)["label"].tolist()
        )
        collect(fig_operating_curves(
            curves, out_dir=fig_dir, tables_dir=tbl_dir, highlight=tuple(top[:5]),
        ))

    ds_name = next(iter(datasets))
    seed = cfg.seeds[0]
    pr = _pr_data(runs, datasets[ds_name], scores_dir, ds_name, seed)
    if pr:
        collect(fig_pr_curves(pr, datasets[ds_name].prevalence, out_dir=fig_dir, tables_dir=tbl_dir))

    collect(fig_encoder_head_heatmap(
        runs, metric=metric, out_dir=fig_dir, tables_dir=tbl_dir,
        eta2=axis_importance(runs, metric=metric),
    ))
    collect(fig_metric_inflation(runs, out_dir=fig_dir, tables_dir=tbl_dir))
    collect(fig_seed_variance(runs, metric=metric, out_dir=fig_dir, tables_dir=tbl_dir))
    collect(fig_pareto(runs, metric=metric, out_dir=fig_dir, tables_dir=tbl_dir))

    traces = _event_traces(runs, datasets[ds_name], scores_dir, ds_name, seed,
                           alarm_budget=cfg.alarm_budget_per_hour,
                           step_min=float(datasets[ds_name].meta.get('step_min', 0.5)),
                           persistence=cfg.persistence)
    if not traces.empty:
        collect(fig_event_timeline(traces, out_dir=fig_dir, tables_dir=tbl_dir))

    collect(fig_spatial_prior_contribution(runs, metric=metric, out_dir=fig_dir, tables_dir=tbl_dir))
    return produced


# ------------------------------------------------------------------- сквозной
def run_experiment(cfg: ExperimentConfig, *, verbose: bool = True) -> dict:
    """Полный цикл: данные, сетка, фигуры, отчёт."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.json").write_text(
        json.dumps(
            {**cfg.__dict__, "seeds": list(cfg.seeds)}, indent=2, ensure_ascii=False, default=str
        ),
        encoding="utf-8",
    )

    if verbose:
        print(f"[1/4] Сборка датасетов: {', '.join(cfg.datasets)}")
    datasets = build_datasets(cfg)
    for name, d in datasets.items():
        if verbose:
            print(
                f"      {name}: узлов {d.n_nodes}, окно {d.window}, "
                f"train {len(d.X_train)}, test {len(d.X_test)}, "
                f"событий {len(d.events)}, prevalence {d.prevalence:.4f}"
            )

    configs = get_grid(cfg.grid)
    if verbose:
        print(f"[2/4] Прогон сетки «{cfg.grid}»: {len(configs)} конфигураций × {len(cfg.seeds)} сидов")
    artifacts = run_grid(
        configs, datasets,
        seeds=cfg.seeds, train_cfg=cfg.train_config(), param_budget=cfg.param_budget,
        out_dir=out, alarm_budget_per_hour=cfg.alarm_budget_per_hour,
        half_life_min=cfg.half_life_min,
        persistence=cfg.persistence, verbose=verbose,
    )
    runs, curves = artifacts["runs"], artifacts["curves"]
    if runs.empty:
        raise RuntimeError("ни один прогон не завершился успешно — см. reports/failures.csv")

    if verbose:
        print("[3/4] Фигуры")
    figures = make_figures(cfg, runs, curves, datasets)

    if verbose:
        print("[4/4] Отчёт")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    results = write_results_md(
        runs, curves, out_path=out / "RESULTS.md",
        manifest=manifest, figures=figures, metric=cfg.primary_metric,
    )
    if verbose:
        print(f"\nГотово: {results}")
        print(f"Фигур: {len(figures)} в {out / 'figures'}")
    return {**artifacts, "figures": figures, "results_md": results, "datasets": datasets}
