#!/usr/bin/env python3
"""Пересобрать RESULTS.md готового прогона текущим кодом отчёта, без обучения.

В отличие от ``make_figures.py`` датасеты и сырые score не нужны: читаются
только сохранённые таблицы каталога прогона — ``runs.csv``, ``curves.csv``,
``manifest.json``, ``experiment_config.json``. Числа не пересчитываются;
меняется лишь то, что делает с ними ``stad.report`` (блокировки выводов,
пометки). Фигуры, которым хватает этих таблиц (01, 02, 04, 05, 06, 07, 09),
перерисовываются; 03 и 08 требуют score и пропускаются.

Каталог прогона не изменяется: результат пишется в ``--out``.

    python scripts/rebuild_report.py --run-dir reports/ft_aed_cv --out reports/ft_aed_cv_v1_rebuilt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stad.experiment import ExperimentConfig, make_figures  # noqa: E402
from stad.report import write_results_md  # noqa: E402


def _code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _shown(path: Path) -> str:
    """Путь для отчёта: относительно корня репозитория, чтобы в закоммиченный
    RESULTS.md не попадало устройство локальной файловой системы."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config(run_dir: Path, out_dir: Path) -> ExperimentConfig:
    """Конфиг прогона из ``experiment_config.json`` с переопределённым ``out_dir``."""
    raw = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    known = set(ExperimentConfig.__dataclass_fields__)
    cfg = ExperimentConfig(**{k: v for k, v in raw.items() if k in known})
    cfg.seeds = tuple(cfg.seeds)
    cfg.out_dir = str(out_dir)
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="каталог прогона: runs.csv, curves.csv, manifest.json, experiment_config.json")
    p.add_argument("--out", required=True, type=Path, help="куда писать RESULTS.md, figures/, tables/")
    p.add_argument("--metric", default=None,
                   help="основная метрика (по умолчанию — primary_metric прогона)")
    args = p.parse_args(argv)

    run_dir, out = args.run_dir, args.out
    if out.resolve() == run_dir.resolve():
        print("--out совпадает с --run-dir: каталог прогона не перезаписывается", file=sys.stderr)
        return 2
    missing = [n for n in ("runs.csv", "manifest.json", "experiment_config.json")
               if not (run_dir / n).exists()]
    if missing:
        print(f"В {run_dir} нет: {', '.join(missing)}", file=sys.stderr)
        return 2

    cfg = load_config(run_dir, out)
    if args.metric is not None:
        cfg.primary_metric = args.metric
    runs = pd.read_csv(run_dir / "runs.csv")
    curves = pd.read_csv(run_dir / "curves.csv") if (run_dir / "curves.csv").exists() else pd.DataFrame()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    out.mkdir(parents=True, exist_ok=True)
    print("Отрисовка фигур (без 03 и 08: им нужны score)…")
    figures = make_figures(cfg, runs, curves, None)

    run_sha = manifest.get("environment", {}).get("git_sha", "?")
    note = (
        f"Пересобрано из `{_shown(run_dir)}` (прогон коммита `{run_sha}`) кодом коммита "
        f"`{_code_sha()}`; числа не пересчитывались, изменены только блокировки выводов "
        f"(ТЗ v2, задача 1.3)"
    )
    results = write_results_md(
        runs, curves, out_path=out / "RESULTS.md",
        manifest=manifest, figures=figures, metric=cfg.primary_metric, source_note=note,
    )
    print(f"Готово: {results}  ({len(figures)} фигур)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
