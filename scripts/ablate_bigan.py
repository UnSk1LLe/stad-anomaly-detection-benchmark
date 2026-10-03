#!/usr/bin/env python3
"""Этап 5: абляция BiGAN на сохранённых чекпойнтах — cycle против critic против combined.

    python scripts/ablate_bigan.py --config configs/ft_aed_cv.yaml --device cuda

Нужны чекпойнты ``cand_gcngru_bigan`` из ``<run-dir>/checkpoints/`` (пишутся
основным прогоном) и те же данные, на которых они обучались. Правило вывода — в
``stad.ablation`` и зафиксировано до просмотра чисел; оно нужно правилу R5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from stad.ablation import ablate_bigan, summarise_ablation  # noqa: E402
from stad.experiment import ExperimentConfig, build_datasets  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-dir", type=Path, help="по умолчанию out_dir конфига")
    p.add_argument("--out-dir", type=Path, help="по умолчанию <run-dir>/ablation")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    run_dir = args.run_dir or Path(cfg.out_dir)
    out_dir = args.out_dir or run_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds) if args.seeds else cfg.seeds

    print("Сборка датасетов…")
    datasets = build_datasets(cfg)
    df = ablate_bigan(
        run_dir, datasets, seeds,
        alarm_budget_per_hour=cfg.alarm_budget_per_hour, half_life_min=cfg.half_life_min,
        persistence=cfg.persistence, node_reduce=cfg.node_reduce, device=args.device,
    )
    random_padf = None
    runs_csv = run_dir / "runs.csv"
    if runs_csv.exists():
        runs = pd.read_csv(runs_csv)
        r = runs.loc[runs["config"] == "ctrl_random", "padf"]
        random_padf = float(r.mean()) if len(r) else None

    table, verdict = summarise_ablation(df, random_padf=random_padf)
    df.to_csv(out_dir / "bigan_ablation_runs.csv", index=False, encoding="utf-8")
    table.to_csv(out_dir / "bigan_ablation_pairs.csv", index=False, encoding="utf-8")
    (out_dir / "BIGAN_ABLATION.md").write_text(
        "# Абляция BiGAN: cycle против critic\n\n" + verdict + "\n", encoding="utf-8"
    )
    print(df.groupby("component")[["padf", "event_recall", "average_precision"]].mean().to_string())
    print()
    print(table.to_string(index=False))
    print("\n" + verdict)
    print(f"\nОтчёт: {out_dir / 'BIGAN_ABLATION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
