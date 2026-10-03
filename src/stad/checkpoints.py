"""Сохранение и загрузка обученных детекторов.

Зачем это отдельный модуль, а не три строки в тренере. Чекпойнт без
метаданных бесполезен: чтобы восстановить модель, нужно знать, каким
энкодером и какой головой она была, с каким ``hidden``, на скольких
узлах и при каком размере окна. Эти значения подбираются на лету
(``budget.match_budget``), и угадать их потом нельзя.

Поэтому вместе с весами сохраняется полный рецепт сборки, статистики
нормализации данных и штамп прогона. Загрузка восстанавливает модель
одним вызовом, без обращения к конфигу.

Чекпойнты не версионируются в git: они тяжёлые и воспроизводятся из
кода. Но они нужны, чтобы:

* посмотреть, что именно выучила модель (веса внимания, матрицу графа);
* прогнать её на новых днях без переобучения;
* использовать детектор как часть контура устранения;
* сравнить модель до и после аугментации на одних и тех же входах.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import Detector, build_detector

FORMAT_VERSION = 1


def save_detector(
    detector: Detector,
    path: str | Path,
    *,
    encoder: str,
    head: str,
    hidden: int,
    n_features: int,
    n_nodes: int,
    window: int,
    encoder_kwargs: dict | None = None,
    head_kwargs: dict | None = None,
    adjacency: np.ndarray | None = None,
    scaler_mean: list[float] | None = None,
    scaler_std: list[float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Сохранить веса вместе с рецептом сборки.

    Матрица смежности кладётся в чекпойнт, а не берётся из датасета при
    загрузке: граф — часть обученной модели (внимание училось именно под
    эту структуру), и подстановка другой смежности тихо изменила бы
    поведение. То же касается статистик нормализации: модель обучена на
    конкретной шкале, и применять её к данным, нормированным иначе,
    нельзя.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "state_dict": detector.state_dict(),
            "recipe": {
                "encoder": encoder,
                "head": head,
                "hidden": int(hidden),
                "n_features": int(n_features),
                "n_nodes": int(n_nodes),
                "window": int(window),
                "encoder_kwargs": dict(encoder_kwargs or {}),
                "head_kwargs": dict(head_kwargs or {}),
            },
            "adjacency": None if adjacency is None else np.asarray(adjacency, dtype=np.float32),
            "scaler": {"mean": scaler_mean, "std": scaler_std},
            "extra": dict(extra or {}),
        },
        path,
    )
    return path


def load_detector(path: str | Path, *, device: str = "cpu") -> tuple[Detector, dict[str, Any]]:
    """Восстановить модель и вернуть её вместе с метаданными.

    Returns
    -------
    detector:
        Готовая к ``score`` модель в режиме ``eval``, с установленным графом.
    meta:
        Рецепт, статистики нормализации и штамп прогона.
    """
    path = Path(path)
    blob = torch.load(path, map_location=device, weights_only=False)
    version = blob.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path.name}: версия формата {version}, ожидается {FORMAT_VERSION}. "
            "Чекпойнт создан другой версией кода — переобучите или напишите миграцию."
        )

    r = blob["recipe"]
    det = build_detector(
        r["encoder"], r["head"], hidden=r["hidden"], n_features=r["n_features"],
        n_nodes=r["n_nodes"], window=r["window"],
        encoder_kwargs=r["encoder_kwargs"], head_kwargs=r["head_kwargs"],
    )
    if blob.get("adjacency") is not None:
        det.set_graph(torch.as_tensor(blob["adjacency"], device=device))
    det.load_state_dict(blob["state_dict"])
    det.to(device).eval()

    meta = {"recipe": r, "scaler": blob.get("scaler", {}), "extra": blob.get("extra", {})}
    return det, meta


def checkpoint_path(out_dir: str | Path, config: str, dataset: str, seed: int) -> Path:
    """Единое соглашение об именовании: ``<config>__<dataset>__seed<k>.pt``.

    Совпадает с именованием сохранённых score, поэтому веса и их выход
    однозначно сопоставимы.
    """
    return Path(out_dir) / "checkpoints" / f"{config}__{dataset}__seed{seed}.pt"


def describe(path: str | Path) -> dict[str, Any]:
    """Прочитать метаданные чекпойнта, не собирая модель."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    n_params = sum(int(np.prod(t.shape)) for t in blob["state_dict"].values())
    return {
        "file": str(path),
        "format_version": blob.get("format_version"),
        "n_params": n_params,
        "size_mb": round(Path(path).stat().st_size / 1e6, 2),
        **blob["recipe"],
        **blob.get("extra", {}),
    }
