#!/usr/bin/env python3
"""Диагностика протокола готового прогона по сохранённым score — без обучения (ТЗ v2, задача 1.1).

Считает: единицы бюджета тревог (ячейки, окна, эпизоды) на тесте и в калибровке,
нулевое распределение случайного контроля и полосу среднего, ORACLE-диагностику
равной реальной частоты тревог, безмодельные score, псевдорепликацию сидов и
парные сравнения с референсом (блоки, фолды, кластерный bootstrap по событиям).

    python scripts/diagnose_run.py --config configs/ft_aed_cv.yaml \\
           --run-dir reports/ft_aed_cv --out reports/ft_aed_cv_diagnostics --tz-reference

Нужен ``runs.csv`` прогона; ``manifest.json``, ``events.csv`` и ``scores/`` — по
наличию. Если ``scores/`` нет, ``--recompute-nontrainable`` пересчитывает клетки
небучаемых конфигураций (контроли, бейзлайны, необученная сеть) на CPU; клетки
обучаемых перечисляются в отчёте как недостающие. Каталог прогона не меняется:
всё пишется в ``--out``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stad.diagnostics import run_diagnostics  # noqa: E402
from stad.experiment import ExperimentConfig, build_datasets  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="конфиг прогона, давшего runs.csv")
    p.add_argument("--run-dir", type=Path, help="каталог прогона (по умолчанию out_dir конфига)")
    p.add_argument("--out", required=True, type=Path, help="куда писать DIAGNOSTICS.md и CSV")
    p.add_argument("--random-seeds", type=int, default=200, help="сидов случайного контроля на фолд")
    p.add_argument("--n-boot", type=int, default=10_000, help="повторов bootstrap")
    p.add_argument("--oracle-rates", type=float, nargs="+", default=[0.25, 1.0],
                   help="ORACLE: частоты ложных ячеек/ч на тесте (только диагностика)")
    p.add_argument("--recompute-nontrainable", action="store_true",
                   help="пересчитать score небучаемых клеток, если их нет в scores/")
    p.add_argument("--tz-reference", action="store_true",
                   help="добавить сверку с числами docs/TZ_PROTOCOL_V2.md §1")
    p.add_argument("--datasets", nargs="+", help="подмножество фолдов (по умолчанию все из конфига)")
    p.add_argument("--device", default="cpu", help="устройство для пересчёта необученной сети")
    args = p.parse_args()

    if not args.config.exists():
        print(f"Конфиг не найден: {args.config}", file=sys.stderr)
        return 2
    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = args.run_dir or Path(cfg.out_dir)
    if not (run_dir / "runs.csv").exists():
        print(f"Нет {run_dir / 'runs.csv'}: каталог не похож на прогон.", file=sys.stderr)
        return 2
    if args.random_seeds < 1:
        print("--random-seeds должен быть ≥ 1", file=sys.stderr)
        return 2
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(cfg.datasets))
        if unknown:
            print(f"Фолдов нет в конфиге: {', '.join(unknown)}", file=sys.stderr)
            return 2
        cfg.datasets = {k: v for k, v in cfg.datasets.items() if k in args.datasets}
    if not (run_dir / "scores").exists():
        print(f"Нет {run_dir / 'scores'}: score не сохранены, часть диагностики будет недоступна"
              + (" (небучаемые клетки будут пересчитаны)." if args.recompute_nontrainable else "."))

    print("Сборка датасетов…", flush=True)
    datasets = build_datasets(cfg)
    path = run_diagnostics(
        cfg, datasets, run_dir, args.out,
        config_path=str(args.config), random_seeds=args.random_seeds, n_boot=args.n_boot,
        oracle_rates=tuple(args.oracle_rates), recompute_nontrainable=args.recompute_nontrainable,
        tz_reference=args.tz_reference, device=args.device, log=lambda m: print(m, flush=True),
    )
    print(f"\nОтчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
