#!/usr/bin/env python3
"""Запуск бенчмарка: данные → сетка → фигуры → отчёт.

Примеры
-------
Быстрая проверка пайплайна на синтетике (минуты)::

    python scripts/run_benchmark.py --config configs/smoke.yaml

Полная сетка на синтетике — отладка и факторные эксперименты::

    python scripts/run_benchmark.py --config configs/synthetic_full.yaml

Итоговый прогон для диссертации на FT-AED с реальными метками::

    python scripts/download_data.py --dataset ft-aed
    python scripts/run_benchmark.py --config configs/ft_aed_core.yaml

Переопределение отдельных полей без правки конфига::

    python scripts/run_benchmark.py --config configs/smoke.yaml --seeds 0 1 2 3 4 --grid core
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stad.experiment import ExperimentConfig, run_experiment  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="YAML-конфиг прогона")
    p.add_argument("--grid", help="переопределить сетку: smoke | core | extended | all")
    p.add_argument("--seeds", type=int, nargs="+", help="переопределить сиды")
    p.add_argument("--out-dir", type=Path, help="переопределить каталог результатов")
    p.add_argument("--param-budget", type=int, help="переопределить бюджет параметров")
    p.add_argument("--epochs", type=int, help="переопределить число эпох")
    p.add_argument("--device", help="cpu | cuda | mps | auto")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="восстановить готовые клетки из out-dir (score + чекпойнт) без переобучения")
    args = p.parse_args()

    if not args.config.exists():
        print(f"Конфиг не найден: {args.config}", file=sys.stderr)
        return 2

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.grid:
        cfg.grid = args.grid
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    if args.out_dir:
        cfg.out_dir = str(args.out_dir)
    if args.param_budget:
        cfg.param_budget = args.param_budget
    if args.epochs:
        cfg.train["epochs"] = args.epochs
    if args.device:
        cfg.train["device"] = args.device

    try:
        run_experiment(cfg, verbose=not args.quiet, resume=args.resume)
    except Exception as exc:
        print(f"\nПрогон прерван: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
