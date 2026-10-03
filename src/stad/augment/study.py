"""Факторный эксперимент: помогает ли синтетика детекции редких аномалий.

Вопрос, на который отвечает модуль: **когда расширение датасета
синтетическими редкими аномалиями реально улучшает детекцию, и зависит
ли ответ от архитектуры энкодера.** В литературе результаты
противоречивы без системного объяснения: LSTM-WGAN-GP даёт +13% F1,
cGAN в i-DREAMS — ноль. Эксперимент разводит возможные причины.

Факторы
-------
``encoder``
    Те же энкодеры, что в основной сетке. Отвечает на «какая архитектура
    выигрывает от аугментации».
``source``
    ``none`` — только реальные аномалии (база сравнения);
    ``gan`` — WGAN-GP на остатках, обученный на реальных аномалиях;
    ``lwr`` — **generator-independent** инъекция по закону сохранения,
    без обученных параметров.
``ratio``
    Сколько синтетических аномальных окон добавляется на одно реальное.

Зачем нужен ``lwr``. Это и есть защита от пробела «циркулярность
оценки». Возможны три исхода, и каждый означает своё:

* прирост у ``gan`` и у ``lwr`` сопоставим → помогает сам факт
  расширения редкого класса, а не выученное распределение; вклад
  работы — не GAN, а протокол аугментации;
* прирост только у ``gan`` → нужно отдельно доказывать, что это не
  запоминание артефактов генератора (кросс-проверка на другом
  генераторе, иначе утверждение не защищается);
* прироста нет ни у кого → публикуемый негативный результат,
  объясняющий противоречие в литературе.

Оценка всегда на **реальных** событиях тестовых дней. Синтетика живёт
только в обучающей выборке — это неснимаемое условие.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.types import SplitData
from ..metrics import eval_segments, evaluation_view, make_calibration
from ..metrics.event_level import event_level_report
from ..metrics.pointwise import average_precision
from .gan import AnomalyGAN, GANConfig
from .injection import LWRShockInjector
from .supervised import SupervisedConfig, SupervisedDetector


@dataclass
class StudyConfig:
    """Спецификация факторного эксперимента."""

    encoders: tuple[str, ...] = ("gcn_gru", "gat_lstm", "transformer")
    sources: tuple[str, ...] = ("none", "lwr", "gan")
    ratios: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    seeds: tuple[int, ...] = (0, 1)
    alarm_budget_per_hour: float = 0.25
    half_life_min: float = 15.0
    persistence: int = 3
    supervised: SupervisedConfig = None  # type: ignore[assignment]
    gan: GANConfig = None                # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.supervised is None:
            self.supervised = SupervisedConfig()
        if self.gan is None:
            self.gan = GANConfig()


def _augment(
    source: str,
    ratio: float,
    data: SplitData,
    gan: AnomalyGAN | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Сгенерировать синтетические аномальные окна поверх реальной нормы."""
    n_real = len(data.X_train_anomalous) if data.X_train_anomalous is not None else 0
    n_syn = int(round(ratio * n_real))
    if source == "none" or n_syn == 0:
        return np.empty((0, *data.X_train.shape[1:]), dtype=np.float32)

    # базой служат РЕАЛЬНЫЕ нормальные окна: синтетическим становится
    # только сигнатура аномалии, а контекст остаётся настоящим
    pick = rng.choice(len(data.X_train), size=n_syn, replace=n_syn > len(data.X_train))
    base = data.X_train[pick]

    if source == "lwr":
        inj = LWRShockInjector(
            n_lanes=int(data.meta.get("lanes", 4)),
            speed_idx=data.feature_names.index("speed"),
            occ_idx=data.feature_names.index("occupancy"),
            vol_idx=data.feature_names.index("volume"),
        )
        out, _ = inj.inject(base, rng)
        return out
    if source == "gan":
        if gan is None:
            raise RuntimeError("для source='gan' нужен обученный AnomalyGAN")
        out, _ = gan.inject(base, rng)
        return out
    raise ValueError(f"неизвестный источник синтетики: {source!r}")


def run_study(
    datasets: dict[str, SplitData],
    cfg: StudyConfig | None = None,
    *,
    out_dir: str | Path = "reports/augmentation",
    verbose: bool = True,
) -> pd.DataFrame:
    """Прогнать факторный эксперимент и сохранить таблицу результатов."""
    cfg = cfg or StudyConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = (
        len(datasets) * len(cfg.seeds) * len(cfg.encoders)
        * (1 + (len(cfg.sources) - 1) * max(1, len([r for r in cfg.ratios if r > 0])))
    )
    done = 0

    for ds_name, data in datasets.items():
        if data.X_train_anomalous is None or len(data.X_train_anomalous) < 8:
            if verbose:
                print(f"[{ds_name}] пропуск: аномальных обучающих окон "
                      f"{0 if data.X_train_anomalous is None else len(data.X_train_anomalous)} (< 8)")
            continue

        step_min = float(data.meta.get("step_min", 0.5))

        for seed in cfg.seeds:
            rng = np.random.default_rng(seed)

            gan = None
            fidelity: dict[str, float] = {}
            if "gan" in cfg.sources and any(r > 0 for r in cfg.ratios):
                if verbose:
                    print(f"[{ds_name} seed={seed}] обучение GAN на "
                          f"{len(data.X_train_anomalous)} реальных аномальных окнах")
                gan = AnomalyGAN(
                    data.n_nodes, data.window, data.n_features, cfg.gan
                ).fit(data.X_train, data.X_train_anomalous, seed=seed, verbose=verbose)
                fidelity = gan.fidelity_report(data.X_train_anomalous, data.X_train, rng)
                if verbose:
                    print(f"    правдоподобие: std_ratio={fidelity['std_ratio']:.2f}, "
                          f"зазор средних={fidelity['mean_abs_gap']:.3f}")

            for encoder in cfg.encoders:
                combos = [("none", 0.0)] + [
                    (s, r) for s in cfg.sources if s != "none" for r in cfg.ratios if r > 0
                ]
                for source, ratio in combos:
                    done += 1
                    tag = f"[{done}/{total}] {ds_name} seed={seed} {encoder} {source} r={ratio}"
                    try:
                        syn = _augment(source, ratio, data, gan, rng)
                        X_anom = (
                            np.concatenate([data.X_train_anomalous, syn])
                            if len(syn) else data.X_train_anomalous
                        )
                        det = SupervisedDetector(
                            encoder, data.n_features, data.n_nodes, data.window, cfg.supervised,
                            encoder_kwargs={"n_lanes": int(data.meta.get("lanes", 4))}
                            if encoder == "hypergraph" else None,
                        )
                        det.set_graph(data.A)
                        det.fit(
                            data.X_train, X_anom,
                            X_val_norm=data.X_val, X_val_anom=data.X_val_anomalous,
                            seed=seed,
                        )
                        scores = det.score(data.X_test)
                        s, y, eid = evaluation_view(scores, data)
                        # порог — по нормальным окнам валидации, не по тестовым меткам
                        rep = event_level_report(
                            s, y, eid, data.t_test, data.events,
                            alarm_budget_per_hour=cfg.alarm_budget_per_hour,
                            half_life_min=cfg.half_life_min,
                            step_min=step_min, persistence=cfg.persistence,
                            calibration=make_calibration(det.score(data.X_val), data),
                            segments=eval_segments(data),
                        )
                        rows.append({
                            "dataset": ds_name, "block": f"{ds_name}|s{seed}", "seed": seed,
                            "encoder": encoder, "source": source, "ratio": ratio,
                            "n_real_anomalous": int(len(data.X_train_anomalous)),
                            "n_synthetic": int(len(syn)),
                            "n_params": det.n_params,
                            "average_precision": average_precision(s, y),
                            **{k: v for k, v in rep.items()},
                            **{f"gan_{k}": v for k, v in fidelity.items()},
                        })
                        if verbose:
                            print(f"{tag}  padf={rep['padf']:.3f} recall={rep['event_recall']:.2f}")
                    except Exception as exc:
                        if verbose:
                            print(f"{tag}  ОШИБКА: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "augmentation_runs.csv", index=False, encoding="utf-8")
        (out / "study_config.json").write_text(
            pd.Series({
                **{k: str(v) for k, v in asdict(cfg).items()},
            }).to_json(indent=2, force_ascii=False),
            encoding="utf-8",
        )
    return df


def summarise(df: pd.DataFrame, *, metric: str = "padf") -> pd.DataFrame:
    """Эффект аугментации относительно базы ``source='none'`` того же энкодера.

    Сравнение парное внутри блока и энкодера: разница берётся к базе,
    обученной на тех же данных тем же сидом. Иначе в «эффект синтетики»
    попадёт разброс инициализации, который на таких выборках
    сопоставим с самим эффектом.
    """
    if df.empty:
        return df
    base = (
        df[df["source"] == "none"]
        .groupby(["block", "encoder"], as_index=False)[metric].mean()
        .rename(columns={metric: "base"})
    )
    m = df[df["source"] != "none"].merge(base, on=["block", "encoder"], how="left")
    m["delta"] = m[metric] - m["base"]
    return (
        m.groupby(["encoder", "source", "ratio"], as_index=False)
        .agg(delta_mean=("delta", "mean"), delta_std=("delta", "std"),
             value=(metric, "mean"), base=("base", "mean"), n=("delta", "size"))
        .sort_values("delta_mean", ascending=False)
    )
