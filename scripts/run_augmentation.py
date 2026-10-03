#!/usr/bin/env python3
"""Факторный эксперимент: помогает ли синтетика детекции редких аномалий.

Отвечает на два вопроса сразу:

1. Улучшает ли расширение обучающей выборки синтетическими редкими
   аномалиями качество детекции на **реальных** событиях.
2. Зависит ли ответ от архитектуры энкодера.

Ключевая защита от пробела «циркулярность оценки»: помимо GAN в сетку
включён generator-independent инжектор на законе сохранения LWR. Если
прирост одинаков у обоих — помогает сам факт расширения редкого класса,
а не выученное распределение.

Примеры
-------
Быстрая проверка пайплайна на одном фолде::

    python scripts/run_augmentation.py --folds 0 --encoders gcn_gru \\
           --ratios 1.0 --seeds 0 --gan-epochs 30 --epochs 5

Полный факторный прогон (для GPU)::

    python scripts/run_augmentation.py --folds 0 1 2 3 \\
           --encoders gcn_gru gat_lstm transformer \\
           --ratios 0.5 1.0 2.0 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from stad.augment import GANConfig, StudyConfig, SupervisedConfig, run_study, summarise  # noqa: E402
from stad.data import load_ft_aed  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/data/ft_aed.yaml")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--encoders", nargs="+", default=["gcn_gru", "gat_lstm", "transformer"])
    p.add_argument("--sources", nargs="+", default=["none", "lwr", "gan"])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=20, help="эпох supervised-обучения")
    p.add_argument("--gan-epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--alarm-budget", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports/augmentation")
    args = p.parse_args()

    kwargs = yaml.safe_load(args.data_config.read_text(encoding="utf-8")) or {}
    kwargs.pop("train_days", None)
    kwargs.pop("val_days", None)
    if not Path(kwargs.get("root", "data/ft-aed")).exists():
        print(f"Нет данных в {kwargs.get('root')}. Сначала: "
              f"python scripts/download_data.py --dataset ft-aed", file=sys.stderr)
        return 2

    print(f"Сборка фолдов: {args.folds}")
    datasets = {}
    for f in args.folds:
        d = load_ft_aed(fold=f, **kwargs)
        datasets[f"fold{f}"] = d
        print(f"  fold{f}: норма {len(d.X_train)}, аномальных {d.meta['n_train_anomalous']}, "
              f"тест {len(d.X_test)} окон / {d.meta['n_events_test']} событий")

    cfg = StudyConfig(
        encoders=tuple(args.encoders),
        sources=tuple(args.sources),
        ratios=tuple([0.0] + [r for r in args.ratios if r > 0]),
        seeds=tuple(args.seeds),
        alarm_budget_per_hour=args.alarm_budget,
        supervised=SupervisedConfig(epochs=args.epochs, hidden=args.hidden, device=args.device),
        gan=GANConfig(epochs=args.gan_epochs, device=args.device),
    )

    df = run_study(datasets, cfg, out_dir=args.out_dir)
    if df.empty:
        print("Ни один прогон не завершился успешно", file=sys.stderr)
        return 1

    summary = summarise(df)
    summary.to_csv(args.out_dir / "augmentation_summary.csv", index=False, encoding="utf-8")
    print("\n=== Эффект аугментации (Δ padf относительно обучения без синтетики) ===")
    print(summary.to_string(index=False))
    print(f"\nПодробности: {args.out_dir / 'augmentation_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
