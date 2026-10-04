#!/usr/bin/env python3
"""Этап 2: чувствительность к агрегации по узлам и к окну меток — без переобучения.

Нужны сохранённые score теста И КАЛИБРОВКИ (``scores/*.npy`` и ``*__calib.npy``):
порог калибруется на валидации, поэтому score старых прогонов, где сохранялся
только тест, для этого не годятся — их придётся получить новым прогоном.

    python scripts/sensitivity.py --config configs/ft_aed_cv.yaml \\
           --run-dir reports/ft_aed_cv_prelim --out-dir reports/sensitivity

Критерий выбора агрегации — положение случайного контроля, а НЕ качество лучшей
модели (см. ``stad.sensitivity.choose_reducer``). Скрипт ничего не пишет в
конфиги: выбор вносится в ``node_reduce`` вручную и ДО итогового прогона.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stad.experiment import ExperimentConfig, build_datasets  # noqa: E402
from stad.registry import get_grid  # noqa: E402
from stad.sensitivity import (  # noqa: E402
    BASE_WINDOW,
    LABEL_WINDOWS,
    NODE_REDUCERS,
    choose_reducer,
    evaluate_saved,
    invariance_diagnostic,
    ranking_stability,
    relabel_test,
    write_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="конфиг прогона, давшего score")
    p.add_argument("--run-dir", type=Path, help="каталог прогона (по умолчанию out_dir конфига)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports/sensitivity")
    p.add_argument("--seeds", type=int, nargs="+", help="сиды, для которых есть score")
    p.add_argument("--reduce", choices=NODE_REDUCERS,
                   help="агрегация для проверки окна меток (по умолчанию — выбранная в п. 1)")
    p.add_argument("--skip-aggregation", action="store_true")
    p.add_argument("--skip-label-window", action="store_true")
    args = p.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = args.run_dir or Path(cfg.out_dir)
    scores_dir = run_dir / "scores"
    seeds = tuple(args.seeds) if args.seeds else cfg.seeds
    if not scores_dir.exists():
        print(f"Нет {scores_dir}: score не сохранены. Сначала прогон.", file=sys.stderr)
        return 2

    print("Сборка датасетов…")
    datasets = build_datasets(cfg)
    configs = [c.name for c in get_grid(cfg.grid)]
    common = dict(
        alarm_budget_per_hour=cfg.alarm_budget_per_hour, half_life_min=cfg.half_life_min,
        persistence=cfg.persistence,
    )

    agg = agg_runs = win_summary = win_runs = None
    chosen = args.reduce or cfg.node_reduce
    n_blocks = len(datasets) * len(seeds)

    if not args.skip_aggregation:
        print("\n[2.1] Агрегация score по узлам:", ", ".join(NODE_REDUCERS))
        agg_runs = {}
        for red in NODE_REDUCERS:
            runs, missing = evaluate_saved(configs, datasets, seeds, scores_dir, reduce=red, **common)
            if runs.empty:
                print(f"Нет ни одного score с валидацией в {scores_dir} (пропущено {len(missing)}).",
                      file=sys.stderr)
                return 2
            agg_runs[red] = runs
        agg = choose_reducer(agg_runs)
        chosen = args.reduce or agg.chosen
        t = agg.table[["reduce", "random_padf", "n_beaten_by_random", "n_trained",
                       "random_below_all_trained", "trained_mean_padf"]]
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nВыбрано: {agg.chosen}. {agg.reason}")
        print(invariance_diagnostic(agg.table))

    if not args.skip_label_window:
        print(f"\n[2.2] Окно меток {list(LABEL_WINDOWS)}; агрегация: {chosen}")
        win_runs = {}
        for win in LABEL_WINDOWS:
            runs, _ = evaluate_saved(
                configs, datasets, seeds, scores_dir, reduce=chosen,
                transform=lambda d, w=win: relabel_test(d, *w), **common,
            )
            win_runs[win] = runs
        if BASE_WINDOW not in win_runs:
            print("BASE_WINDOW отсутствует среди LABEL_WINDOWS", file=sys.stderr)
            return 2
        win_summary = ranking_stability(win_runs)
        print(win_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    path = write_report(
        args.out_dir, agg=agg, agg_runs=agg_runs, win_summary=win_summary, win_runs=win_runs,
        meta={"run_dir": str(run_dir), "n_seeds": len(seeds), "n_blocks": n_blocks},
    )
    print(f"\nОтчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
