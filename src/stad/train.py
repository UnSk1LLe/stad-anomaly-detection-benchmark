"""Обучение и скоринг одной конфигурации.

Три решения, зафиксированные здесь, одинаковы для всех архитектур —
иначе сравнение превращается в сравнение расписаний обучения:

1. **Ранняя остановка по валидационной потере на нормальных окнах.**
   Валидация не содержит аномалий и не содержит меток, поэтому выбор
   чекпойнта не подглядывает в тест. Выбор чекпойнта по тестовой
   метрике — самая частая скрытая утечка в литературе.
2. **Один и тот же оптимизатор, расписание и число эпох-бюджет.**
   Различаться могут только те гиперпараметры, которые физически
   неприменимы к другой архитектуре.
3. **Фиксированные сиды, никакого best-of-N.** Сид задаётся снаружи;
   агрегация по сидам — среднее, не максимум (Lyu, 2026).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data.types import SplitData
from .model import Detector


@dataclass
class TrainConfig:
    """Единое расписание обучения для всех конфигураций."""

    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 6
    d_steps: int = 1          # шагов критика на шаг генератора (adversarial)
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 0        # 0 = молча

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


@dataclass
class TrainOutcome:
    """Что вернуло обучение, помимо самих score."""

    scores: np.ndarray
    best_val_loss: float
    epochs_run: int
    train_seconds: float
    inference_ms_per_window: float
    n_params: int
    history: list[dict[str, float]] = field(default_factory=list)
    extras: dict[str, float] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """Детерминизм в пределах, которые даёт PyTorch на CPU/GPU."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(X: np.ndarray, batch_size: int, *, shuffle: bool, workers: int = 0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def train_detector(
    detector: Detector,
    data: SplitData,
    cfg: TrainConfig,
    *,
    seed: int = 0,
    randomize_only: bool = False,
) -> TrainOutcome:
    """Обучить детектор и вернуть score на тесте.

    Parameters
    ----------
    randomize_only:
        Если True, обучение **пропускается**: модель остаётся со
        случайной инициализацией. Это реализация контроля
        «untrained» из Kim et al. (2021): разность с обученной версией
        измеряет реальный вклад обучения.
    """
    set_seed(seed)
    device = cfg.resolve_device()
    detector = detector.to(device)
    detector.set_graph(torch.from_numpy(data.A).to(device))

    history: list[dict[str, float]] = []
    best_val, best_state, epochs_run = float("inf"), None, 0
    t0 = time.perf_counter()

    if not randomize_only:
        train_dl = _loader(data.X_train, cfg.batch_size, shuffle=True, workers=cfg.num_workers)
        val_dl = _loader(data.X_val, cfg.batch_size, shuffle=False, workers=cfg.num_workers)

        opt = torch.optim.AdamW(
            list(detector.main_parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        opt_d = None
        if detector.adversarial:
            d_params = list(detector.head.d_parameters())
            if d_params:
                opt_d = torch.optim.AdamW(d_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

        stale = 0
        for epoch in range(cfg.epochs):
            detector.train()
            running = 0.0
            n_batches = 0
            for (xb,) in train_dl:
                xb = xb.to(device, non_blocking=True)

                if opt_d is not None:
                    for _ in range(cfg.d_steps):
                        opt_d.zero_grad(set_to_none=True)
                        ld = detector.d_loss(xb)
                        ld.backward()
                        nn.utils.clip_grad_norm_(list(detector.head.d_parameters()), cfg.grad_clip)
                        opt_d.step()

                opt.zero_grad(set_to_none=True)
                loss = detector.loss(xb)
                loss.backward()
                nn.utils.clip_grad_norm_(list(detector.main_parameters()), cfg.grad_clip)
                opt.step()
                running += float(loss.detach())
                n_batches += 1

            sched.step()
            detector.eval()
            with torch.no_grad():
                val_losses = [float(detector.loss(xb.to(device))) for (xb,) in val_dl]
            val = float(np.mean(val_losses)) if val_losses else float("nan")
            history.append({"epoch": epoch, "train_loss": running / max(1, n_batches), "val_loss": val})
            epochs_run = epoch + 1

            if cfg.log_every and (epoch % cfg.log_every == 0):
                print(f"  epoch {epoch:3d}  train {history[-1]['train_loss']:.5f}  val {val:.5f}")

            # ранняя остановка по валидации на нормальных окнах, не по тесту
            if np.isfinite(val) and val < best_val - 1e-6:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in detector.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            detector.load_state_dict(best_state)

    train_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------- скоринг
    detector.eval()
    test_dl = _loader(data.X_test, cfg.batch_size, shuffle=False, workers=cfg.num_workers)
    chunks: list[np.ndarray] = []
    t_inf = time.perf_counter()
    with torch.no_grad():
        for (xb,) in test_dl:
            chunks.append(detector.score(xb.to(device)).float().cpu().numpy())
    inference_s = time.perf_counter() - t_inf
    scores = np.concatenate(chunks, axis=0).astype(np.float32)

    if scores.shape != data.y_test.shape:
        raise RuntimeError(f"score {scores.shape} != y_test {data.y_test.shape}")
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.nanmax(scores[np.isfinite(scores)], initial=0.0))
        print(f"  ВНИМАНИЕ: {n_bad} нечисловых значений в score заменены")

    extras: dict[str, float] = {}
    if hasattr(detector.head, "correction_share"):
        with torch.no_grad():
            xb = torch.from_numpy(data.X_test[: cfg.batch_size]).to(device)
            x_enc, x_tgt = detector._split(xb)
            extras["physics_correction_share"] = float(
                detector.head.correction_share(detector.encoder(x_enc), x_tgt)
            )

    return TrainOutcome(
        scores=scores,
        best_val_loss=best_val if np.isfinite(best_val) else float("nan"),
        epochs_run=epochs_run,
        train_seconds=train_seconds,
        inference_ms_per_window=1000.0 * inference_s / max(1, len(data.X_test)),
        n_params=detector.n_params,
        history=history,
        extras=extras,
    )
