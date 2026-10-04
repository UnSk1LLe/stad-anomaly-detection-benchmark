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

import sys
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
    progress: bool = False    # полоса прогресса по эпохам с оценкой оставшегося времени

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
    #: score калибровочных окон (дни валидации) — для калибровки порога, меток не нужно.
    calib_scores: np.ndarray | None = None


def set_seed(seed: int) -> None:
    """Детерминизм в пределах, которые даёт PyTorch на CPU/GPU."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    """``95 -> '1м35с'``, ``3700 -> '1ч01м'``."""
    s = max(0, int(round(seconds)))
    if s >= 3600:
        return f"{s // 3600}ч{(s % 3600) // 60:02d}м"
    if s >= 60:
        return f"{s // 60}м{s % 60:02d}с"
    return f"{s}с"


def progress_line(
    epoch: int, total: int, train_loss: float, val_loss: float, elapsed: float,
    *, stale: int = 0, patience: int = 0, width: int = 20,
) -> str:
    """Строка прогресса после эпохи ``epoch`` (с нуля).

    Оставшееся время — оценка сверху: ранняя остановка может прервать
    обучение раньше, но заранее неизвестно когда, поэтому считается до ``epochs``.
    """
    done = epoch + 1
    filled = int(width * done / max(1, total))
    per_epoch = elapsed / done
    eta = per_epoch * (total - done)
    stop = f"  ранняя остановка {stale}/{patience}" if patience and stale else ""
    return (
        f"  [{'#' * filled}{'-' * (width - filled)}] эпоха {done}/{total}  "
        f"train {train_loss:.4f}  val {val_loss:.4f}  {per_epoch:.1f}с/эп  "
        f"ост. ≤{format_duration(eta)}{stop}"
    )


class _EpochProgress:
    """В терминале — одна перезаписываемая строка; в файле лога — редкие строки."""

    def __init__(self, enabled: bool, total: int) -> None:
        self.enabled = enabled
        self.tty = enabled and sys.stdout.isatty()
        self.every = max(1, total // 8)
        self.total = total
        self._width = 0

    def update(self, epoch: int, line: str) -> None:
        if not self.enabled:
            return
        if self.tty:
            self._width = max(self._width, len(line))
            print("\r" + line.ljust(self._width), end="", flush=True)
        elif (epoch + 1) % self.every == 0 or epoch + 1 == self.total:
            print(line, flush=True)

    def close(self, epochs_run: int, stopped_early: bool) -> None:
        if not self.enabled:
            return
        if self.tty:
            print("\r" + " " * self._width + "\r", end="", flush=True)
        if stopped_early:
            print(f"  ранняя остановка на эпохе {epochs_run}/{self.total}", flush=True)


def _loader(X: np.ndarray, batch_size: int, *, shuffle: bool, workers: int = 0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)


def _score_windows(
    detector: Detector, X: np.ndarray, cfg: TrainConfig, device: torch.device
) -> tuple[np.ndarray, float]:
    """Score окон батчами: ``(score [n, N], секунд)``. Нечисловые значения заменяются явно."""
    dl = _loader(X, cfg.batch_size, shuffle=False, workers=cfg.num_workers)
    chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for (xb,) in dl:
            chunks.append(detector.score(xb.to(device)).float().cpu().numpy())
    seconds = time.perf_counter() - t0
    scores = np.concatenate(chunks, axis=0).astype(np.float32)
    if not np.isfinite(scores).all():
        n_bad = int((~np.isfinite(scores)).sum())
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.nanmax(scores[np.isfinite(scores)], initial=0.0))
        print(f"  ВНИМАНИЕ: {n_bad} нечисловых значений в score заменены")
    return scores, seconds


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
        bar = _EpochProgress(cfg.progress, cfg.epochs)
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
            improved = bool(np.isfinite(val) and val < best_val - 1e-6)
            if improved:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in detector.state_dict().items()}
                stale = 0
            else:
                stale += 1
            bar.update(epoch, progress_line(
                epoch, cfg.epochs, history[-1]["train_loss"], val, time.perf_counter() - t0,
                stale=stale, patience=cfg.patience,
            ))
            if not improved and stale >= cfg.patience:
                break
        bar.close(epochs_run, stopped_early=epochs_run < cfg.epochs)

        if best_state is not None:
            detector.load_state_dict(best_state)

    train_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------- скоринг
    detector.eval()
    scores, inference_s = _score_windows(detector, data.X_test, cfg, device)
    if scores.shape != data.y_test.shape:
        raise RuntimeError(f"score {scores.shape} != y_test {data.y_test.shape}")

    # калибровочная выборка (окна дней валидации без размеченных окон) скорится тем
    # же детектором: по ней калибруется порог. Меток она не несёт, тест порогу не нужен.
    calib_scores = None
    if data.X_calib is not None:
        calib_scores, _ = _score_windows(detector, data.X_calib, cfg, device)

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
        calib_scores=calib_scores,
    )
