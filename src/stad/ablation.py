"""Абляция BiGAN: что даёт критик, а что цикл-реконструкция.

``BiGANHead.score_components`` отдаёт ``cycle`` и ``critic`` по отдельности. Здесь
обе части прогоняются как самостоятельные детекторы на СОХРАНЁННЫХ чекпойнтах (без
переобучения) и сравниваются с полным score при том же протоколе: порог по
валидации, та же агрегация по узлам, те же блоки.

Правило вывода зафиксировано до просмотра чисел (оно нужно правилу R5):

* критик НЕСЁТ информацию об аномалии, только если полный score значимо лучше
  ``cycle`` по ``padf`` (парный bootstrap по блокам, p < 0.05, Δ = combined − cycle > 0);
* иначе вклад даёт цикл-реконструкция, а критик ничего не добавляет — прямое
  эмпирическое подтверждение теоретического пробела ``D* = 1/2``: у оптимального
  критика выход равен 1/2 при любом входе, и информации об аномалии в нём нет.

Оговорка: «не значимо лучше» при малом числе блоков может означать недостаток
мощности, а не отсутствие вклада; сводка печатает число блоков и доверительный интервал.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import checkpoint_path, load_detector
from .data.types import SplitData
from .metrics import full_report
from .metrics.stats import paired_bootstrap_test

COMPONENTS: tuple[str, ...] = ("cycle", "critic", "combined")


@torch.no_grad()
def bigan_components(
    detector, X: np.ndarray, *, batch_size: int = 64, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Три score BiGAN-головы для окон ``X [n, N, T, F]``: каждый ``[n, N]``."""
    if not hasattr(detector.head, "score_components"):
        raise TypeError(f"голова {type(detector.head).__name__} не BiGAN: нет score_components")
    detector.eval()
    parts: dict[str, list[np.ndarray]] = {k: [] for k in COMPONENTS}
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(device)
        h = detector.encode(xb)
        for k, v in detector.head.score_components(h).items():
            parts[k].append(v.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in parts.items()}


def ablate_bigan(
    run_dir: str | Path,
    datasets: dict[str, SplitData],
    seeds: tuple[int, ...],
    *,
    alarm_budget_per_hour: float,
    half_life_min: float,
    persistence: int,
    node_reduce: str = "max",
    config_name: str = "cand_gcngru_bigan",
    device: str = "cpu",
    check_saved: bool = True,
    atol: float = 1e-3,
) -> pd.DataFrame:
    """Метрики каждой компоненты по каждому блоку ``датасет × сид``.

    ``check_saved``: пересчитанный ``combined`` обязан совпасть с сохранённым в
    ``scores/*.npy`` — иначе чекпойнт применён к иначе подготовленным данным, и вся
    абляция недействительна.
    """
    run_dir = Path(run_dir)
    rows: list[dict] = []
    for ds_name, data in datasets.items():
        for seed in seeds:
            ckpt = checkpoint_path(run_dir, config_name, ds_name, seed)
            if not ckpt.exists():
                raise FileNotFoundError(f"нет чекпойнта {ckpt}")
            det, _ = load_detector(ckpt, device=device)
            test = bigan_components(det, data.X_test, device=device)
            calib = bigan_components(det, data.X_calib, device=device)

            if check_saved:
                saved = run_dir / "scores" / f"{config_name}__{ds_name}__seed{seed}.npy"
                if saved.exists():
                    err = float(np.max(np.abs(np.load(saved) - test["combined"])))
                    if err > atol * max(1.0, float(np.max(np.abs(test["combined"])))):
                        raise RuntimeError(
                            f"{ckpt.name}: пересчитанный combined расходится с сохранённым "
                            f"(max|Δ|={err:.3g}) — данные или нормализация не те"
                        )

            for comp in COMPONENTS:
                rep = full_report(
                    test[comp], data, calib_scores=calib[comp],
                    alarm_budget_per_hour=alarm_budget_per_hour, half_life_min=half_life_min,
                    persistence=persistence, reduce=node_reduce,
                )
                rows.append({
                    "component": comp, "dataset": ds_name, "seed": seed,
                    "block": f"{ds_name}|s{seed}",
                    **{k: rep[k] for k in ("padf", "event_recall", "median_delay_min",
                                           "average_precision", "alarms_per_hour")},
                })
    return pd.DataFrame(rows)


def summarise_ablation(df: pd.DataFrame, *, random_padf: float | None = None) -> tuple[pd.DataFrame, str]:
    """Сводка по компонентам и вывод по зафиксированному правилу."""
    pivot = df.pivot_table(index="block", columns="component", values="padf", aggfunc="mean")
    n_blocks = len(pivot)
    cmp_rows = []
    for a, b in (("combined", "cycle"), ("combined", "critic"), ("cycle", "critic")):
        res = paired_bootstrap_test(pivot[a].to_numpy(), pivot[b].to_numpy())
        cmp_rows.append({"a": a, "b": b, "delta_padf": res["mean_diff"],
                         "ci_lo": res.get("ci_lo", np.nan), "ci_hi": res.get("ci_hi", np.nan),
                         "p_value": res["p_value"], "n_blocks": n_blocks})
    table = pd.DataFrame(cmp_rows)

    means = df.groupby("component")[["padf", "event_recall", "average_precision"]].mean()
    c = table[(table["a"] == "combined") & (table["b"] == "cycle")].iloc[0]
    critic_informative = bool(c["p_value"] < 0.05 and c["delta_padf"] > 0)
    parts = [
        f"padf: cycle {means.loc['cycle', 'padf']:.3f}, critic {means.loc['critic', 'padf']:.3f}, "
        f"combined {means.loc['combined', 'padf']:.3f}"
        + (f", случайный контроль {random_padf:.3f}" if random_padf is not None else "")
        + f" ({n_blocks} блоков)."
    ]
    if critic_informative:
        parts.append(
            f"Полный score значимо лучше cycle (Δ={c['delta_padf']:+.4f}, "
            f"95% ДИ [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], p={c['p_value']:.3f}): критик несёт "
            "информацию об аномалии сверх цикл-реконструкции. R5: подтверждается линия CCB-GraphGAN."
        )
    else:
        parts.append(
            f"Полный score НЕ значимо лучше cycle (Δ={c['delta_padf']:+.4f}, "
            f"95% ДИ [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], p={c['p_value']:.3f}): вклад даёт "
            "цикл-реконструкция, критик ничего не добавляет. Это эмпирическое подтверждение "
            "теоретического пробела D*=1/2 (R5)."
            + (" ВНИМАНИЕ: блоков мало, это может быть недостаток мощности." if n_blocks < 10 else "")
        )
    return table, " ".join(parts)
