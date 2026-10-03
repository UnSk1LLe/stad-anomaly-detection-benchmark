#!/usr/bin/env python3
"""Получение данных FT-AED и проверка схемы.

Данные не входят в репозиторий: лицензия и размер. Скрипт клонирует
официальный репозиторий данных, затем **проверяет** соответствие схемы
тому, что ожидает загрузчик, и печатает готовый блок сопоставления
столбцов для ``configs/data/ft_aed.yaml``.

Почему схема не угадывается кодом: структура файлов в апстриме может
меняться, и молчаливое угадывание столбца «занятость» — прямой путь к
бессмысленным результатам, которые при этом выглядят правдоподобно.

    python scripts/download_data.py --dataset ft-aed
    python scripts/download_data.py --dataset ft-aed --check-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "ft-aed": {
        "repo": "https://github.com/acoursey3/freeway-anomaly-data.git",
        # Файл хранится в Git LFS. Обычный `git clone` отдаёт только
        # указатель на 134 байта — проверено. Поэтому качаем напрямую
        # с media-эндпоинта, минуя LFS-протокол.
        "direct": (
            "https://media.githubusercontent.com/media/acoursey3/"
            "freeway-anomaly-data/main/nashville_freeway_anomaly.csv"
        ),
        "filename": "nashville_freeway_anomaly.csv",
        "expected_bytes": 123_234_268,
        "site": "https://acoursey3.github.io/ft-aed/",
        "paper": "https://arxiv.org/abs/2406.15283",
        "baselines": "https://github.com/acoursey3/freeway-anomaly-detection",
        "license": "см. LICENSE в репозитории данных",
        "note": (
            "I-24, ~18 миль, 49 постов × 4 полосы, радар, шаг 30 с, будние дни октября 2023, "
            ">3.7 млн измерений; метки — отчёты Nashville TMC + ручная разметка."
        ),
    }
}

# Фактическая схема официального файла (широкий формат по полосам,
# метки встроены). Проверена на скачанных данных, не угадана.
EXPECTED_FLOW = (
    "day", "unix_time", "milemarker",
    "lane1_speed", "lane1_volume", "lane1_occ",
    "lane4_speed", "lane4_volume", "lane4_occ",
)
EXPECTED_INC = ("human_label", "crash_record")


def download(dataset: str, dest: Path) -> int:
    """Скачать файл напрямую: LFS через клон не работает.

    Клонирование репозитория данных отдаёт LFS-указатель (134 байта), а
    не сам CSV: `git lfs pull` требует учётных данных, которых у
    анонимного доступа нет. Прямая ссылка на media-эндпоинт отдаёт
    настоящие 118 МБ.
    """
    spec = SOURCES[dataset]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / spec["filename"]

    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"Файл уже скачан: {target} ({target.stat().st_size / 1e6:.0f} МБ)")
        return 0

    if shutil.which("curl") is None:
        print(f"curl не найден. Скачайте вручную:\n  {spec['direct']}\nи положите в {target}",
              file=sys.stderr)
        return 1

    print(f"Загрузка {spec['direct']}\n  -> {target}")
    r = subprocess.run(["curl", "-sSL", "--fail", "-o", str(target), spec["direct"]])
    if r.returncode != 0:
        print(f"Загрузка не удалась. Скачайте вручную: {spec['direct']}", file=sys.stderr)
        return r.returncode

    size = target.stat().st_size
    if size < 1_000_000:
        head = target.read_bytes()[:64]
        print(f"Получено {size} байт — похоже на LFS-указатель, а не данные:\n  {head!r}",
              file=sys.stderr)
        return 1
    expected = spec.get("expected_bytes")
    note = "" if not expected else (
        " (совпадает с ожидаемым)" if size == expected else f" (ожидалось {expected})"
    )
    print(f"Скачано {size / 1e6:.0f} МБ{note}")
    return 0


def _legacy_clone(dataset: str, dest: Path) -> int:
    spec = SOURCES[dataset]
    if dest.exists() and any(dest.iterdir()):
        print(f"Каталог уже существует и не пуст: {dest}")
        return 0
    if shutil.which("git") is None:
        print("git не найден в PATH", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Клонирование {spec['repo']} → {dest}")
    r = subprocess.run(["git", "clone", "--depth", "1", spec["repo"], str(dest)])
    if r.returncode != 0:
        print(
            "Клонирование не удалось. Скачайте вручную:\n"
            f"  сайт:      {spec['site']}\n"
            f"  статья:    {spec['paper']}\n"
            f"  бейзлайны: {spec['baselines']}\n"
            f"и распакуйте в {dest}",
            file=sys.stderr,
        )
    return r.returncode


def inspect(dest: Path) -> int:
    """Показать найденные файлы и столбцы, предложить блок сопоставления."""
    try:
        import pandas as pd
    except ImportError:
        print("Нужен pandas: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if not dest.exists():
        print(f"Каталог не найден: {dest}", file=sys.stderr)
        return 2

    tabular = [
        p for p in dest.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".pq", ".gz"}
    ]
    if not tabular:
        print(f"В {dest} не найдено табличных файлов (.csv/.parquet)", file=sys.stderr)
        return 2

    print(f"\nНайдено табличных файлов: {len(tabular)}")
    for p in sorted(tabular)[:25]:
        size_mb = p.stat().st_size / 1e6
        try:
            head = (
                pd.read_parquet(p).head(3)
                if p.suffix.lower() in {".parquet", ".pq"}
                else pd.read_csv(p, nrows=3)
            )
            cols = list(head.columns)
        except Exception as exc:
            print(f"  {p.relative_to(dest)}  ({size_mb:.1f} МБ)  — не прочитан: {exc}")
            continue
        print(f"  {p.relative_to(dest)}  ({size_mb:.1f} МБ)")
        print(f"      столбцы: {cols}")

        missing_flow = [c for c in EXPECTED_FLOW if c not in cols]
        missing_inc = [c for c in EXPECTED_INC if c not in cols]
        if not missing_flow and not missing_inc:
            print("      → полностью соответствует ожидаемой схеме")
        elif missing_flow:
            print(f"      → не хватает столбцов измерений: {missing_flow}")
        if missing_inc:
            print(f"      → не хватает столбцов меток: {missing_inc}")

    print(
        "\nДальше:\n"
        "  1. Если имена столбцов отличаются от ожидаемых, пропишите соответствие\n"
        "     в configs/data/ft_aed.yaml, ключ `columns`.\n"
        "  2. Проверьте загрузку:\n"
        "     python -c \"import sys; sys.path.insert(0,'src'); "
        "from stad.data import load_ft_aed; d=load_ft_aed('data/ft-aed'); "
        "print(d.meta['n_events_total'], d.meta['n_nodes'], round(d.prevalence,3))\"\n"
        "     Ожидается: 63 события, 196 узлов.\n"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ft-aed", choices=sorted(SOURCES))
    p.add_argument("--dest", type=Path, help="куда скачивать (по умолчанию data/<dataset>)")
    p.add_argument("--check-only", action="store_true", help="не скачивать, только проверить схему")
    args = p.parse_args()

    dest = args.dest or (DATA / args.dataset)
    spec = SOURCES[args.dataset]
    print(f"{args.dataset}: {spec['note']}")
    print(f"Лицензия: {spec['license']}")

    if not args.check_only:
        rc = download(args.dataset, dest)
        if rc != 0:
            return rc
    return inspect(dest)


if __name__ == "__main__":
    raise SystemExit(main())
