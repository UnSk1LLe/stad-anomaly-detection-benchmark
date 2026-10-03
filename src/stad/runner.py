"""Прогон сетки конфигураций и сбор артефактов.

Что считается артефактом одного прогона:

* строка метрик в ``runs.csv`` (одна строка = конфигурация × сид × датасет);
* операционная кривая в ``curves.csv``;
* сырые score в ``scores/<config>__<dataset>__seed<k>.npy`` — чтобы
  фигуры и статистику можно было пересчитать без повторного обучения;
* score валидации там же, с суффиксом ``__val.npy``: порог калибруется по
  ним, а не по тесту, и без них порог не пересчитать;
* таблица подбора бюджета в ``budget.csv``.

Блок (``block``) — это «датасет × сид»: единица, внутри которой методы
сравниваются напрямую в тесте Фридмана. Он же обеспечивает, что
агрегация идёт по сидам усреднением, а не выбором лучшего.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import build_baseline
from .budget import match_budget
from .checkpoints import checkpoint_path, save_detector
from .data.types import SplitData
from .encoders import ENCODER_LABELS
from .heads import HEAD_LABELS, HEAD_MECHANISM
from .metrics import (
    eval_segments,
    evaluation_view,
    full_report,
    make_calibration,
    operating_curve,
)
from .model import build_detector
from .registry import Config, GROUP_LABELS
from .train import TrainConfig, format_duration, train_detector


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_stamp() -> dict[str, str]:
    """Штамп окружения — обязательная часть воспроизводимости."""
    import torch

    return {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda or "cpu",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def estimate_remaining(
    remaining: list[Config],
    durations: dict[str, float],
    trainable: dict[str, bool],
) -> float:
    """Оценка оставшихся секунд по средним временам уже выполненных клеток.

    Время клетки берётся как среднее по той же конфигурации; для конфигурации,
    которой ещё не было, — среднее по клеткам того же рода (обучаемые отдельно
    от небучаемых), иначе дешёвые бейзлайны занижали бы оценку. Данные
    поздних фолдов больше, чем ранних, поэтому оценка слегка оптимистична.
    """
    by_name: dict[str, list[float]] = {}
    for name, sec in durations.items():
        by_name.setdefault(name, []).append(sec)
    kind: dict[bool, list[float]] = {True: [], False: []}
    for name, secs in by_name.items():
        kind[trainable[name]].extend(secs)
    total = 0.0
    for cfg in remaining:
        own = by_name.get(cfg.name)
        pool = own or kind[cfg.is_trainable]
        total += sum(pool) / len(pool) if pool else 0.0
    return total


def run_config(
    cfg: Config,
    data: SplitData,
    *,
    seed: int,
    dataset_name: str,
    train_cfg: TrainConfig,
    param_budget: int,
    alarm_budget_per_hour: float = 1.0,
    half_life_min: float = 15.0,
    persistence: int = 3,
    node_reduce: str = "max",
    checkpoint_dir: str | Path | None = None,
) -> tuple[dict, pd.DataFrame, np.ndarray, dict | None, np.ndarray]:
    """Выполнить одну конфигурацию.

    Возвращает ``(метрики, кривая, score теста, бюджет, score валидации)``.
    Порог калибруется по score валидации и переносится на тест без изменений.
    """
    budget_row: dict | None = None

    if cfg.is_trainable:
        head_kwargs = dict(cfg.head_kwargs)
        enc_kwargs = dict(cfg.encoder_kwargs)
        # физической голове нужно знать число полос и индексы признаков
        if cfg.head == "physics":
            head_kwargs.setdefault("lanes", int(data.meta.get("lanes", 3)))
            for key, feat in (("occ_idx", "occupancy"), ("vol_idx", "volume")):
                if feat in data.feature_names:
                    head_kwargs.setdefault(key, data.feature_names.index(feat))
        if cfg.encoder == "hypergraph":
            enc_kwargs.setdefault("n_lanes", int(data.meta.get("lanes", 3)))

        br = match_budget(
            cfg.encoder, cfg.head,
            target=param_budget,
            n_features=data.n_features, n_nodes=data.n_nodes, window=data.window,
            encoder_kwargs=enc_kwargs, head_kwargs=head_kwargs,
        )
        budget_row = {
            "config": cfg.name, "dataset": dataset_name, **asdict(br),
            "deviation": br.deviation,
        }
        budget_row.pop("tried", None)

        detector = build_detector(
            cfg.encoder, cfg.head, hidden=br.hidden,
            n_features=data.n_features, n_nodes=data.n_nodes, window=data.window,
            encoder_kwargs=enc_kwargs, head_kwargs=head_kwargs,
        )
        outcome = train_detector(
            detector, data, train_cfg, seed=seed, randomize_only=cfg.randomize_only
        )
        scores, val_scores = outcome.scores, outcome.val_scores

        # веса сохраняются вместе с рецептом сборки: hidden подбирается на
        # лету под бюджет параметров, и без рецепта чекпойнт не восстановить
        if checkpoint_dir is not None and not cfg.randomize_only:
            save_detector(
                detector,
                checkpoint_path(checkpoint_dir, cfg.name, dataset_name, seed),
                encoder=cfg.encoder, head=cfg.head, hidden=br.hidden,
                n_features=data.n_features, n_nodes=data.n_nodes, window=data.window,
                encoder_kwargs=enc_kwargs, head_kwargs=head_kwargs,
                adjacency=data.A,
                scaler_mean=data.meta.get("scaler_mean"),
                scaler_std=data.meta.get("scaler_std"),
                extra={
                    "config": cfg.name, "dataset": dataset_name, "seed": seed,
                    "epochs_run": outcome.epochs_run,
                    "best_val_loss": outcome.best_val_loss,
                    "step_min": data.meta.get("step_min"),
                    "label_source": data.meta.get("label_source"),
                },
            )
        runtime = {
            "n_params": outcome.n_params,
            "hidden": br.hidden,
            "budget_within_tolerance": br.within_tolerance,
            "epochs_run": outcome.epochs_run,
            "train_seconds": outcome.train_seconds,
            "inference_ms_per_window": outcome.inference_ms_per_window,
            "best_val_loss": outcome.best_val_loss,
            "uses_graph": bool(detector.uses_graph),
            **outcome.extras,
        }
    else:
        kwargs = dict(cfg.baseline_kwargs)
        if cfg.baseline in {"random", "iforest"}:
            kwargs.setdefault("seed", seed)
        if cfg.baseline == "california":
            kwargs.setdefault("lanes", int(data.meta.get("lanes", 3)))
            if "occupancy" in data.feature_names:
                kwargs.setdefault("occ_idx", data.feature_names.index("occupancy"))

        import time

        model = build_baseline(cfg.baseline, **kwargs)
        t0 = time.perf_counter()
        model.fit(data)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        scores = np.asarray(model.score(data), dtype=np.float32)
        val_scores = np.asarray(model.score_val(data), dtype=np.float32)
        inf_s = time.perf_counter() - t1
        runtime = {
            "n_params": int(getattr(model, "n_params", 0)),
            "hidden": np.nan,
            "budget_within_tolerance": True,
            "epochs_run": 0,
            "train_seconds": fit_s,
            "inference_ms_per_window": 1000.0 * inf_s / max(1, len(data.X_test)),
            "best_val_loss": np.nan,
            "uses_graph": cfg.baseline == "california",
        }

    metrics = full_report(scores, data, val_scores=val_scores,
                          alarm_budget_per_hour=alarm_budget_per_hour,
                          half_life_min=half_life_min, persistence=persistence,
                          reduce=node_reduce)
    s_eval, y_eval, eid_eval = evaluation_view(scores, data, reduce=node_reduce)
    curve = operating_curve(
        s_eval, y_eval, eid_eval, data.t_test, data.events,
        half_life_min=half_life_min, step_min=float(data.meta.get("step_min", 0.5)),
        persistence=persistence,
        calibration=make_calibration(val_scores, data, reduce=node_reduce),
        segments=eval_segments(data),
    )

    row = {
        "config": cfg.name,
        "label": cfg.label,
        "group": cfg.group,
        "group_label": GROUP_LABELS.get(cfg.group, cfg.group),
        "encoder": cfg.encoder,
        "head": cfg.head,
        "encoder_label": ENCODER_LABELS.get(cfg.encoder) if cfg.encoder else None,
        "head_label": HEAD_LABELS.get(cfg.head) if cfg.head else None,
        "mechanism": HEAD_MECHANISM.get(cfg.head) if cfg.head else "none",
        "baseline": cfg.baseline,
        "dataset": dataset_name,
        "seed": seed,
        "block": f"{dataset_name}|s{seed}",
        "rationale": cfg.rationale,
        **runtime,
        **metrics,
    }
    curve = curve.assign(config=cfg.name, label=cfg.label, dataset=dataset_name, seed=seed)
    return row, curve, scores, budget_row, val_scores


def _eta_text(t_grid: float, remaining: list[Config], cells: list[tuple[str, float]],
              trainable: dict[str, bool]) -> str:
    elapsed = time.perf_counter() - t_grid
    if not remaining:
        return f"[прошло {format_duration(elapsed)}]"
    # среднее по клеткам одной конфигурации: для оценки хватает последних значений
    by_cfg: dict[str, list[float]] = {}
    for n, sec in cells:
        by_cfg.setdefault(n, []).append(sec)
    avg = {n: sum(v) / len(v) for n, v in by_cfg.items()}
    eta = estimate_remaining(remaining, avg, trainable)
    return f"[прошло {format_duration(elapsed)}, осталось ≈{format_duration(eta)}]"


def run_grid(
    configs: tuple[Config, ...],
    datasets: dict[str, SplitData],
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_cfg: TrainConfig | None = None,
    param_budget: int = 200_000,
    out_dir: str | Path = "reports",
    alarm_budget_per_hour: float = 1.0,
    half_life_min: float = 15.0,
    persistence: int = 3,
    node_reduce: str = "max",
    save_checkpoints: bool = True,
    save_scores: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Прогнать всю сетку и сохранить артефакты.

    Порядок циклов — датасет → сид → конфигурация — выбран так, чтобы
    внутри блока все методы видели **одни и те же** данные и одну и ту
    же инициализацию генератора: иначе блоковая структура теста
    Фридмана нарушается.
    """
    train_cfg = train_cfg or TrainConfig()
    out_dir = Path(out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    budgets: list[dict] = []
    failures: list[dict] = []

    total = len(datasets) * len(seeds) * len(configs)
    done = 0
    plan = [cfg for _ in datasets for _ in seeds for cfg in configs]
    trainable = {c.name: c.is_trainable for c in configs}
    cell_seconds: list[tuple[str, float]] = []      # (конфигурация, секунд)
    t_grid = time.perf_counter()
    for ds_name, data in datasets.items():
        for seed in seeds:
            for cfg in configs:
                done += 1
                tag = f"[{done}/{total}] {ds_name} seed={seed} {cfg.name}"
                if verbose and cfg.is_trainable:
                    print(f"{tag}  обучение…", flush=True)
                t_cell = time.perf_counter()
                try:
                    row, curve, scores, budget, val_scores = run_config(
                        cfg, data, seed=seed, dataset_name=ds_name,
                        train_cfg=train_cfg, param_budget=param_budget,
                        alarm_budget_per_hour=alarm_budget_per_hour,
                        half_life_min=half_life_min, persistence=persistence,
                        node_reduce=node_reduce,
                        checkpoint_dir=out_dir if save_checkpoints else None,
                    )
                    rows.append(row)
                    curves.append(curve)
                    if budget:
                        budgets.append(budget)
                    if save_scores:
                        np.save(out_dir / "scores" / f"{cfg.name}__{ds_name}__seed{seed}.npy", scores)
                        # score валидации — чтобы пересчитывать порог без переобучения
                        np.save(out_dir / "scores" / f"{cfg.name}__{ds_name}__seed{seed}__val.npy",
                                val_scores)
                    if verbose:
                        cell_seconds.append((cfg.name, time.perf_counter() - t_cell))
                        print(
                            f"{tag}  recall={row['event_recall']:.3f} "
                            f"delay={row['median_delay_min']:+.1f}м padf={row['padf']:.3f} "
                            f"AP={row['average_precision']:.4f}  "
                            f"{_eta_text(t_grid, plan[done:], cell_seconds, trainable)}",
                            flush=True,
                        )
                except Exception as exc:  # прогон одной клетки не должен ронять сетку
                    failures.append({"config": cfg.name, "dataset": ds_name, "seed": seed,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    if verbose:
                        print(f"{tag}  ОШИБКА: {type(exc).__name__}: {exc}")

    runs = pd.DataFrame(rows)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    budget_df = pd.DataFrame(budgets)
    fail_df = pd.DataFrame(failures)

    runs.to_csv(out_dir / "runs.csv", index=False, encoding="utf-8")
    if not curves_df.empty:
        curves_df.to_csv(out_dir / "curves.csv", index=False, encoding="utf-8")
    if not budget_df.empty:
        budget_df.to_csv(out_dir / "budget.csv", index=False, encoding="utf-8")
    if not fail_df.empty:
        fail_df.to_csv(out_dir / "failures.csv", index=False, encoding="utf-8")

    manifest = {
        "environment": environment_stamp(),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "param_budget": param_budget,
        "alarm_budget_per_hour": alarm_budget_per_hour,
        "half_life_min": half_life_min,
        "persistence": persistence,
        "node_reduce": node_reduce,
        "threshold_source": "validation",
        "save_checkpoints": save_checkpoints,
        "train_config": asdict(train_cfg),
        "datasets": {k: v.meta for k, v in datasets.items()},
        "prevalence": {k: v.prevalence for k, v in datasets.items()},
        "n_configs": len(configs),
        "n_failures": len(failures),
        "reporting_rule": (
            "Агрегация по сидам — среднее. Выбор лучшего сида (best-of-N) запрещён: "
            "при best-of-N часть метрик становится обманываемой (Lyu, 2026). "
            "Число сидов раскрыто в этом манифесте."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {"runs": runs, "curves": curves_df, "budget": budget_df, "failures": fail_df}
