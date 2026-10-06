"""Прогон сетки конфигураций и сбор артефактов.

Что считается артефактом одного прогона:

* строка метрик в ``runs.csv`` (одна строка = конфигурация × сид × датасет);
* операционная кривая в ``curves.csv``;
* сырые score в ``scores/<config>__<dataset>__seed<k>.npy`` — чтобы
  фигуры и статистику можно было пересчитать без повторного обучения;
* score калибровочной выборки там же, с суффиксом ``__calib.npy``: порог калибруется по
  ним, а не по тесту, и без них порог не пересчитать;
* таблица подбора бюджета в ``budget.csv``;
* пособытийная таблица в ``events.csv``: строка на событие теста каждой клетки
  (найдено ли, задержка, кредит ``padf``). Среднее кредита по событиям клетки
  равно её ``padf``; таблица нужна статистике, где единица — событие (ТЗ v2,
  задачи 1.2 и 2.4).

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
import torch

from .baselines import build_baseline
from .budget import match_budget
from .checkpoints import checkpoint_path, describe, load_detector, save_detector
from .data.types import SplitData
from .encoders import ENCODER_LABELS
from .eventlog import EVENT_COLUMNS, event_table
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


def _build_recipe(cfg: Config, data: SplitData, param_budget: int):
    """``(encoder_kwargs, head_kwargs, результат подбора бюджета)`` для обучаемой конфигурации."""
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
    return enc_kwargs, head_kwargs, br


def _budget_row(cfg: Config, dataset_name: str, br) -> dict:
    row = {"config": cfg.name, "dataset": dataset_name, **asdict(br), "deviation": br.deviation}
    row.pop("tried", None)
    return row


def _finalize(
    cfg: Config, data: SplitData, scores: np.ndarray, calib_scores: np.ndarray, runtime: dict,
    *, seed: int, dataset_name: str, alarm_budget_per_hour: float, half_life_min: float,
    persistence: int, node_reduce: str,
) -> tuple[dict, pd.DataFrame]:
    """Метрики, операционная кривая и строка ``runs.csv`` по готовым score."""
    metrics = full_report(scores, data, calib_scores=calib_scores,
                          alarm_budget_per_hour=alarm_budget_per_hour,
                          half_life_min=half_life_min, persistence=persistence,
                          reduce=node_reduce)
    s_eval, y_eval, eid_eval = evaluation_view(scores, data, reduce=node_reduce)
    curve = operating_curve(
        s_eval, y_eval, eid_eval, data.t_test, data.events,
        half_life_min=half_life_min, step_min=float(data.meta.get("step_min", 0.5)),
        persistence=persistence,
        calibration=make_calibration(calib_scores, data, reduce=node_reduce),
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
    return row, curve


def resume_cell(
    cfg: Config,
    data: SplitData,
    *,
    seed: int,
    dataset_name: str,
    out_dir: str | Path,
    train_cfg: TrainConfig,
    param_budget: int,
    alarm_budget_per_hour: float = 1.0,
    half_life_min: float = 15.0,
    persistence: int = 3,
    node_reduce: str = "max",
) -> tuple[dict, pd.DataFrame, np.ndarray, dict | None, np.ndarray] | None:
    """Восстановить готовую клетку из сохранённых score и чекпойнта, не переобучая.

    Только для обучаемых конфигураций с чекпойнтом (бейзлайны и необученный
    контроль дёшевы и пересчитываются заново). Возвращает ``None``, если
    клетку восстановить нельзя: тогда она обучается как обычно. Проверки
    против подхвата чужих файлов: формы score, ``hidden`` чекпойнта равен
    подобранному под ТЕКУЩИЙ бюджет, совпадают энкодер, голова, датасет и сид.

    Метрики пересчитываются по тем же score теми же функциями, что и при
    обычном прогоне, поэтому совпадают с ними. Не восстанавливаются время
    обучения и вывода (NaN); в строке ``resumed=True``.
    """
    if not cfg.is_trainable or cfg.randomize_only:
        return None
    out_dir = Path(out_dir)
    base = out_dir / "scores" / f"{cfg.name}__{dataset_name}__seed{seed}"
    f_test, f_calib = Path(f"{base}.npy"), Path(f"{base}__calib.npy")
    ckpt = checkpoint_path(out_dir, cfg.name, dataset_name, seed)
    if not (f_test.exists() and f_calib.exists() and ckpt.exists()):
        return None
    scores, calib_scores = np.load(f_test), np.load(f_calib)
    if data.X_calib is None or scores.shape != data.y_test.shape \
            or calib_scores.shape != (len(data.X_calib), data.n_nodes):
        return None

    enc_kwargs, head_kwargs, br = _build_recipe(cfg, data, param_budget)
    info = describe(ckpt)
    if (info.get("hidden") != br.hidden or info.get("encoder") != cfg.encoder
            or info.get("head") != cfg.head or info.get("config") != cfg.name
            or info.get("dataset") != dataset_name or info.get("seed") != seed):
        return None

    detector, _ = load_detector(ckpt)
    extras: dict[str, float] = {}
    if hasattr(detector.head, "correction_share"):
        with torch.no_grad():
            xb = torch.from_numpy(data.X_test[: train_cfg.batch_size])
            x_enc, x_tgt = detector._split(xb)
            extras["physics_correction_share"] = float(
                detector.head.correction_share(detector.encoder(x_enc), x_tgt)
            )
    runtime = {
        "n_params": detector.n_params,
        "hidden": br.hidden,
        "budget_within_tolerance": br.within_tolerance,
        "epochs_run": info.get("epochs_run", np.nan),
        "train_seconds": np.nan,
        "inference_ms_per_window": np.nan,
        "best_val_loss": info.get("best_val_loss", np.nan),
        "uses_graph": bool(detector.uses_graph),
        "resumed": True,
        **extras,
    }
    row, curve = _finalize(
        cfg, data, scores, calib_scores, runtime, seed=seed, dataset_name=dataset_name,
        alarm_budget_per_hour=alarm_budget_per_hour, half_life_min=half_life_min,
        persistence=persistence, node_reduce=node_reduce,
    )
    return row, curve, scores, _budget_row(cfg, dataset_name, br), calib_scores


def run_fingerprint(
    configs: tuple[Config, ...], datasets: dict[str, SplitData], train_cfg: TrainConfig,
    param_budget: int,
) -> dict:
    """Что должно совпадать, чтобы готовые клетки одного каталога были сравнимы с новыми."""
    tc = asdict(train_cfg)
    for k in ("device", "num_workers", "log_every", "progress"):
        tc.pop(k, None)
    keys = ("label_source", "step_min", "window", "n_train_windows", "n_calib_windows",
            "n_test_windows", "days_train", "days_val", "days_test", "lead_min", "trail_min")
    return json.loads(json.dumps({
        "param_budget": param_budget,
        "train": tc,
        "configs": [c.name for c in configs],
        "datasets": {n: {k: d.meta.get(k) for k in keys} for n, d in datasets.items()},
    }, default=str))


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
        enc_kwargs, head_kwargs, br = _build_recipe(cfg, data, param_budget)
        budget_row = _budget_row(cfg, dataset_name, br)

        detector = build_detector(
            cfg.encoder, cfg.head, hidden=br.hidden,
            n_features=data.n_features, n_nodes=data.n_nodes, window=data.window,
            encoder_kwargs=enc_kwargs, head_kwargs=head_kwargs,
        )
        outcome = train_detector(
            detector, data, train_cfg, seed=seed, randomize_only=cfg.randomize_only
        )
        scores, calib_scores = outcome.scores, outcome.calib_scores

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
            "resumed": False,
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
        calib_scores = np.asarray(model.score_calib(data), dtype=np.float32)
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
            "resumed": False,
        }

    row, curve = _finalize(
        cfg, data, scores, calib_scores, runtime, seed=seed, dataset_name=dataset_name,
        alarm_budget_per_hour=alarm_budget_per_hour, half_life_min=half_life_min,
        persistence=persistence, node_reduce=node_reduce,
    )
    return row, curve, scores, budget_row, calib_scores


#: Колонки ``events.csv``: идентификатор клетки и пособытийная таблица.
EVENTS_CSV_COLUMNS: tuple[str, ...] = ("config", "dataset", "seed", *EVENT_COLUMNS)


def _cell_events(
    row: dict, scores: np.ndarray, data: SplitData, *, half_life_min: float, persistence: int,
    node_reduce: str,
) -> pd.DataFrame:
    """Строки ``events.csv`` одной клетки.

    Порог — откалиброванный на валидации и уже записанный в строку ``runs.csv``;
    здесь он не пересчитывается, тестовые метки на него не влияют. Порог другого
    происхождения (oracle по тесту) — ошибка: в ``events.csv`` его не было бы видно.
    """
    if row.get("threshold_source") != "validation":
        raise ValueError(f"events.csv строится только по порогу валидации, "
                         f"а в строке threshold_source={row.get('threshold_source')!r}")
    ev = event_table(scores, data, float(row["threshold"]), half_life_min=half_life_min,
                     persistence=persistence, node_reduce=node_reduce)
    ev.insert(0, "seed", row["seed"])
    ev.insert(0, "dataset", row["dataset"])
    ev.insert(0, "config", row["config"])
    return ev


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
    resume: bool = False,
) -> dict[str, pd.DataFrame]:
    """Прогнать всю сетку и сохранить артефакты.

    Порядок циклов — датасет → сид → конфигурация — выбран так, чтобы
    внутри блока все методы видели **одни и те же** данные и одну и ту
    же инициализацию генератора: иначе блоковая структура теста
    Фридмана нарушается.

    ``resume``: готовые клетки (score теста и калибровки + чекпойнт) восстанавливаются
    из ``out_dir`` без переобучения (:func:`resume_cell`). Прогон по часам, и
    аварийная перезагрузка не должна стоить всей сетки. Каталог должен
    принадлежать прогону с той же конфигурацией: отпечаток
    ``resume_fingerprint.json`` сверяется, расхождение — ошибка.
    """
    train_cfg = train_cfg or TrainConfig()
    out_dir = Path(out_dir)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)

    fp = run_fingerprint(configs, datasets, train_cfg, param_budget)
    fp_path = out_dir / "resume_fingerprint.json"
    if resume:
        if fp_path.exists():
            old = json.loads(fp_path.read_text(encoding="utf-8"))
            if old != fp:
                diff = sorted(k for k in set(fp) | set(old) if fp.get(k) != old.get(k))
                raise RuntimeError(
                    f"возобновление отклонено: конфигурация прогона изменилась ({', '.join(diff)}). "
                    f"Готовые клетки в {out_dir} несопоставимы с новыми; используйте другой --out-dir."
                )
        elif verbose:
            print("ВНИМАНИЕ: отпечатка конфигурации в каталоге нет (прогон начат до его введения); "
                  "готовые клетки принимаются на доверии, проверяются только hidden и формы score.",
                  flush=True)
    fp_path.write_text(json.dumps(fp, indent=2, ensure_ascii=False), encoding="utf-8")
    n_resumed = 0

    rows: list[dict] = []
    curves: list[pd.DataFrame] = []
    budgets: list[dict] = []
    events: list[pd.DataFrame] = []
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
                if resume:
                    got = None
                    try:
                        got = resume_cell(
                            cfg, data, seed=seed, dataset_name=ds_name, out_dir=out_dir,
                            train_cfg=train_cfg, param_budget=param_budget,
                            alarm_budget_per_hour=alarm_budget_per_hour,
                            half_life_min=half_life_min, persistence=persistence,
                            node_reduce=node_reduce,
                        )
                        if got is not None:
                            # внутри try: сбой таблицы — та же клетка заново, а не падение сетки
                            cell_events = _cell_events(got[0], got[2], data, half_life_min=half_life_min,
                                                       persistence=persistence, node_reduce=node_reduce)
                    except Exception as exc:       # битый файл — просто обучить заново
                        got = None
                        if verbose:
                            print(f"{tag}  восстановить не удалось ({type(exc).__name__}: {exc})",
                                  flush=True)
                    if got is not None:
                        row, curve, _, budget, _ = got
                        rows.append(row)
                        events.append(cell_events)
                        curves.append(curve)
                        budgets.append(budget)
                        n_resumed += 1
                        if verbose:
                            print(f"{tag}  восстановлено: recall={row['event_recall']:.3f} "
                                  f"padf={row['padf']:.3f}", flush=True)
                        continue
                if verbose and cfg.is_trainable:
                    print(f"{tag}  обучение…", flush=True)
                t_cell = time.perf_counter()
                try:
                    row, curve, scores, budget, calib_scores = run_config(
                        cfg, data, seed=seed, dataset_name=ds_name,
                        train_cfg=train_cfg, param_budget=param_budget,
                        alarm_budget_per_hour=alarm_budget_per_hour,
                        half_life_min=half_life_min, persistence=persistence,
                        node_reduce=node_reduce,
                        checkpoint_dir=out_dir if save_checkpoints else None,
                    )
                    # до добавления строки: клетка с ошибкой не оставляет ни строки, ни событий
                    cell_events = _cell_events(row, scores, data, half_life_min=half_life_min,
                                               persistence=persistence, node_reduce=node_reduce)
                    rows.append(row)
                    events.append(cell_events)
                    curves.append(curve)
                    if budget:
                        budgets.append(budget)
                    if save_scores:
                        np.save(out_dir / "scores" / f"{cfg.name}__{ds_name}__seed{seed}.npy", scores)
                        # score валидации — чтобы пересчитывать порог без переобучения
                        np.save(out_dir / "scores" / f"{cfg.name}__{ds_name}__seed{seed}__calib.npy",
                                calib_scores)
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
    events_df = (pd.concat(events, ignore_index=True) if events
                 else pd.DataFrame(columns=list(EVENTS_CSV_COLUMNS)))

    runs.to_csv(out_dir / "runs.csv", index=False, encoding="utf-8")
    events_df.to_csv(out_dir / "events.csv", index=False, encoding="utf-8")
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
        "n_resumed_cells": n_resumed,
        "reporting_rule": (
            "Агрегация по сидам — среднее. Выбор лучшего сида (best-of-N) запрещён: "
            "при best-of-N часть метрик становится обманываемой (Lyu, 2026). "
            "Число сидов раскрыто в этом манифесте."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return {"runs": runs, "curves": curves_df, "budget": budget_df, "failures": fail_df,
            "events": events_df}
