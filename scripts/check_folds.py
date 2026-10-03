#!/usr/bin/env python3
"""Сколько событий в тесте каждого фолда — проверка геометрии CV до прогона.

Шесть фолдов по 2 тестовых дня дают шесть блоков для теста Фридмана, но
каждый фолд при этом тоньше. Если в каком-то фолде меньше ``--min-events``
событий, event-recall квантуется слишком грубо, и от шести фолдов нужно
отказаться (вернуться к четырём по 3 дня). Скрипт только считает и
возвращает код выхода: решение принимает человек, а не скрипт.

Обе разметки проверяются по умолчанию: итоговый прогон идёт по ``crash``
(44 официальных отчёта), а ``both`` (63) нужен для второго прогона.

    python scripts/check_folds.py
    python scripts/check_folds.py --n-folds 4 --fold-test-days 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from stad.data import DataNotDownloaded, load_ft_aed  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/data/ft_aed.yaml")
    p.add_argument("--n-folds", type=int, default=6)
    p.add_argument("--fold-test-days", type=int, default=2)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--label-sources", nargs="+", default=["crash", "both"])
    args = p.parse_args()

    kwargs = yaml.safe_load(args.data_config.read_text(encoding="utf-8")) or {}
    kwargs.pop("train_days", None)
    kwargs.pop("val_days", None)
    kwargs["n_folds"] = args.n_folds
    kwargs["fold_test_days"] = args.fold_test_days

    worst = {}
    for src in args.label_sources:
        print(f"\nlabel_source = {src}   ({args.n_folds} фолдов × {args.fold_test_days} дня)")
        print(f"{'фолд':>5} {'тест-дни':<22} {'событий':>8} {'окон':>8} {'доля аном.':>11} {'val-окон':>9}")
        counts = []
        for f in range(args.n_folds):
            try:
                d = load_ft_aed(fold=f, **{**kwargs, "label_source": src})
            except DataNotDownloaded as exc:
                print(exc, file=sys.stderr)
                return 2
            except ValueError as exc:
                print(f"{f:>5}  ОШИБКА: {exc}")
                counts.append(0)
                continue
            counts.append(d.meta["n_events_test"])
            print(f"{f:>5} {str(d.meta['days_test']):<22} {d.meta['n_events_test']:>8} "
                  f"{d.meta['n_test_windows']:>8} {d.prevalence:>11.3f} {d.meta['n_val_windows']:>9}")
        worst[src] = min(counts)
        print(f"минимум: {worst[src]}, всего: {sum(counts)}")

    bad = {s: w for s, w in worst.items() if w < args.min_events}
    print()
    if bad:
        print("НЕ ХВАТАЕТ событий (порог {}): {}. Вернуться к 4 фолдам по 3 дня."
              .format(args.min_events, ", ".join(f"{s}: min={w}" for s, w in bad.items())))
        return 1
    print(f"Во всех фолдах не меньше {args.min_events} событий.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
