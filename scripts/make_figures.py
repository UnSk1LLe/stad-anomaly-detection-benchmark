#!/usr/bin/env python3
"""Пересобрать фигуры и отчёт из уже сохранённых результатов.

Обучение сетки занимает часы, а фигуры переделываются многократно.
Скрипт читает ``reports/runs.csv``, ``reports/curves.csv`` и сохранённые
score из ``reports/scores/``, заново собирает датасеты по тому же
конфигу (детерминировано) и перерисовывает всё без обучения.

    python scripts/make_figures.py --config configs/ft_aed_core.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stad.experiment import ExperimentConfig, build_datasets, make_figures  # noqa: E402
from stad.report import write_results_md  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, help="переопределить каталог результатов")
    p.add_argument("--metric", help="переопределить основную метрику")
    args = p.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.out_dir:
        cfg.out_dir = str(args.out_dir)
    if args.metric:
        cfg.primary_metric = args.metric

    out = Path(cfg.out_dir)
    runs_path = out / "runs.csv"
    if not runs_path.exists():
        print(f"Нет {runs_path} — сначала выполните run_benchmark.py", file=sys.stderr)
        return 2

    runs = pd.read_csv(runs_path)
    curves = pd.read_csv(out / "curves.csv") if (out / "curves.csv").exists() else pd.DataFrame()

    print("Сборка датасетов для PR-кривых и разбора события…")
    datasets = build_datasets(cfg)

    print("Отрисовка фигур…")
    figures = make_figures(cfg, runs, curves, datasets)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    results = write_results_md(
        runs, curves, out_path=out / "RESULTS.md",
        manifest=manifest, figures=figures, metric=cfg.primary_metric,
    )
    print(f"Готово: {results}  ({len(figures)} фигур)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
