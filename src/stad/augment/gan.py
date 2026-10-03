"""WGAN-GP для синтеза редких аномалий трафика.

Что именно генерируется — и почему не окно целиком. Окно FT-AED имеет
размерность 196 × 8 × 3 = 4704, а реальных аномальных событий в обучении
десятки. Генерировать такой тензор «с нуля» на таких данных означает
гарантированный коллапс мод. Поэтому генерируется **остаток**:

.. math::
    x_{anom} = x_{normal} + \\delta,\\quad \\delta = G(z, c)

Остаток **аддитивный**, а не мультипликативный, потому что данные
приходят z-нормализованными: там значения знакопеременны, и умножение
на :math:`(1+\\delta)` не имеет физического смысла — оно меняло бы знак
возмущения в зависимости от знака значения. В стандартизованной шкале
естественная единица возмущения — сигма, и :math:`\\delta` измеряется
именно в ней. Реальный контекст
(суточный профиль, узкое место, шум детекторов, корреляции между
полосами) остаётся настоящим; синтетической становится только та часть,
которой в данных не хватает. Это и есть «расширение датасета редкими
аномалиями», а не генерация трафика с нуля.

Выбор WGAN-GP, а не ванильного GAN, обусловлен размером выборки:
градиентный штраф даёт устойчивое обучение на десятках событий, где
обычный GAN с BCE расходится. Обзор MTSAD прямо отмечает нестабильность
GAN (исчезающие градиенты) как основное ограничение семейства.

**Архитектурная несхожесть с детектором обязательна.** Этот генератор —
свёрточный по времени и по пространству, критик тоже свёрточный. Голова
``BiGANHead`` в сетке детекторов работает на представлениях энкодера
полносвязными слоями. Общих компонентов нет, иначе прирост от
аугментации нельзя было бы отличить от запоминания артефактов
собственного генератора (пробел «циркулярность оценки»).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GANConfig:
    """Гиперпараметры обучения. Фиксируются до эксперимента."""

    latent: int = 32
    hidden: int = 64
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.9)
    n_critic: int = 5
    gp_weight: float = 10.0
    max_delta: float = 3.0          # предел остатка в единицах sigma
    device: str = "cpu"


class _Generator(nn.Module):
    """``z, c -> delta [N, T, F]`` — мультипликативная сигнатура аномалии.

    Условие ``c`` — агрегат нормального контекста окна (среднее по
    времени на узел), чтобы сигнатура согласовывалась с режимом трафика:
    в свободном потоке и в заторе аномалия выглядит по-разному.
    """

    def __init__(self, n_nodes: int, window: int, n_features: int, cfg: GANConfig) -> None:
        super().__init__()
        self.n_nodes, self.window, self.n_features = n_nodes, window, n_features
        h = cfg.hidden
        self.cond = nn.Sequential(nn.Linear(n_nodes * n_features, h), nn.ReLU(), nn.Linear(h, h))
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent + h, h * 2), nn.ReLU(),
            nn.Linear(h * 2, h * window), nn.ReLU(),
        )
        # разворачиваем по пространству свёрткой: соседние узлы коррелированы
        self.spatial = nn.Sequential(
            nn.Conv1d(h, h, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(h, n_features, kernel_size=5, padding=2),
        )
        self.node_mix = nn.Linear(window, n_nodes)
        self.max_delta = cfg.max_delta
        self.h = h

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if context.dim() == 4:                 # [B, N, T, F] -> [B, N, F]
            context = context.mean(dim=2)
        c = self.cond(context.reshape(B, -1))
        g = self.fc(torch.cat([z, c], -1)).reshape(B, self.h, self.window)
        nodes = torch.relu(self.node_mix(g))                       # [B, h, N]
        delta = self.spatial(nodes)                                # [B, F, N]
        delta = delta.transpose(1, 2).unsqueeze(2)                 # [B, N, 1, F]
        time_shape = torch.linspace(0, 1, self.window, device=z.device).view(1, 1, -1, 1)
        # ramp по времени: аномалия нарастает, а не включается скачком
        return self.max_delta * torch.tanh(delta) * time_shape


class _Critic(nn.Module):
    """Оценивает реалистичность пары «контекст, остаток»."""

    def __init__(self, n_nodes: int, window: int, n_features: int, cfg: GANConfig) -> None:
        super().__init__()
        h = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv2d(2 * n_features, h, kernel_size=(5, 3), padding=(2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(h, h, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(h, 1)

    def forward(self, context: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # [B, N, T, F] -> [B, 2F, N, T]
        x = torch.cat([context, delta], dim=-1).permute(0, 3, 1, 2)
        return self.out(self.net(x).flatten(1))


class AnomalyGAN:
    """WGAN-GP, обучаемый на реальных аномальных окнах.

    Обучающая пара — ``(нормальный контекст, наблюдённый остаток)``, где
    остаток извлекается из реальных аномальных окон относительно
    медианного нормального профиля того же узла. Так генератор учит
    именно «как выглядит аномалия», а не «как выглядит трафик».
    """

    def __init__(self, n_nodes: int, window: int, n_features: int, cfg: GANConfig | None = None) -> None:
        self.cfg = cfg or GANConfig()
        self.device = torch.device(self.cfg.device)
        self.G = _Generator(n_nodes, window, n_features, self.cfg).to(self.device)
        self.D = _Critic(n_nodes, window, n_features, self.cfg).to(self.device)
        self.n_nodes, self.window, self.n_features = n_nodes, window, n_features
        self.history: list[dict[str, float]] = []
        self._fitted = False

    # ------------------------------------------------------------- вспомог.
    @staticmethod
    def _context(x: np.ndarray) -> np.ndarray:
        """Агрегат контекста окна: среднее по времени, ``[B, N, F]``."""
        return x.mean(axis=2)

    def _gradient_penalty(self, ctx: torch.Tensor, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        eps = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
        mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
        score = self.D(ctx, mix)
        grad = torch.autograd.grad(
            outputs=score, inputs=mix, grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True,
        )[0]
        return ((grad.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()

    # ------------------------------------------------------------------ fit
    def fit(self, X_normal: np.ndarray, X_anomalous: np.ndarray, *, seed: int = 0, verbose: bool = False) -> "AnomalyGAN":
        """Обучить на реальных аномальных окнах.

        Остаток считается относительно медианного нормального профиля
        узла: ``delta = x_anom − median_normal``, в единицах sigma
        (данные приходят z-нормализованными). Медиана, а не среднее —
        она устойчива к остаточным выбросам в «нормальной» выборке.
        """
        if len(X_anomalous) < 8:
            raise ValueError(
                f"аномальных окон всего {len(X_anomalous)} — для обучения GAN нужно хотя бы 8. "
                "Расширьте окно разметки или возьмите больше обучающих дней."
            )
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        # опорный профиль нормы: по нему считается наблюдённый остаток
        base = np.median(X_normal, axis=0, keepdims=True).astype(np.float32)   # [1, N, T, F]
        self.base_ = base

        delta = (X_anomalous - base).clip(-self.cfg.max_delta, self.cfg.max_delta)
        ctx = self._context(X_anomalous)
        ctx = np.repeat(ctx[:, :, None, :], self.window, axis=2)       # [B, N, T, F]

        D_t = torch.from_numpy(delta.astype(np.float32)).to(self.device)
        C_t = torch.from_numpy(ctx.astype(np.float32)).to(self.device)

        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)

        n = len(D_t)
        bs = min(self.cfg.batch_size, n)
        for epoch in range(self.cfg.epochs):
            idx = torch.from_numpy(rng.permutation(n)).to(self.device)
            d_loss = g_loss = 0.0
            steps = 0
            for i in range(0, n, bs):
                sel = idx[i : i + bs]
                real, ctx_b = D_t[sel], C_t[sel]
                if len(real) < 2:
                    continue

                for _ in range(self.cfg.n_critic):
                    z = torch.randn(len(real), self.cfg.latent, device=self.device)
                    fake = self.G(z, ctx_b).detach()
                    opt_d.zero_grad(set_to_none=True)
                    loss_d = (
                        self.D(ctx_b, fake).mean() - self.D(ctx_b, real).mean()
                        + self.cfg.gp_weight * self._gradient_penalty(ctx_b, real, fake)
                    )
                    loss_d.backward()
                    opt_d.step()

                z = torch.randn(len(real), self.cfg.latent, device=self.device)
                opt_g.zero_grad(set_to_none=True)
                loss_g = -self.D(ctx_b, self.G(z, ctx_b)).mean()
                loss_g.backward()
                opt_g.step()

                d_loss += float(loss_d.detach()); g_loss += float(loss_g.detach()); steps += 1

            if steps:
                self.history.append({"epoch": epoch, "d_loss": d_loss / steps, "g_loss": g_loss / steps})
                if verbose and epoch % 50 == 0:
                    print(f"  GAN epoch {epoch:4d}  D {d_loss/steps:+.3f}  G {g_loss/steps:+.3f}")

        self._fitted = True
        return self

    # --------------------------------------------------------------- sample
    @torch.no_grad()
    def inject(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Наложить сгенерированную сигнатуру на реальные нормальные окна."""
        if not self._fitted:
            raise RuntimeError("AnomalyGAN не обучен")
        ctx = np.repeat(self._context(X)[:, :, None, :], self.window, axis=2)
        z = torch.randn(len(X), self.cfg.latent, device=self.device)
        delta = self.G(z, torch.from_numpy(ctx.astype(np.float32)).to(self.device)).cpu().numpy()
        out = X + delta
        affected = np.abs(delta).max(axis=(2, 3)) > 0.2
        return out.astype(np.float32), affected

    # ------------------------------------------------------------- качество
    @torch.no_grad()
    def fidelity_report(self, X_anomalous: np.ndarray, X_normal: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        """Простые проверки правдоподобия синтетики.

        Это не метрика качества генерации как таковая, а **санитарные
        проверки**: совпадают ли порядок величины и знак возмущения с
        реальными аномалиями. Если синтетика не похожа даже по этим
        грубым статистикам, обсуждать вклад аугментации рано.
        """
        n = min(len(X_anomalous), len(X_normal))
        fake, _ = self.inject(X_normal[:n], rng)
        real_d = X_anomalous[:n] - self.base_
        fake_d = fake - self.base_
        return {
            "real_delta_mean": float(real_d.mean()),
            "fake_delta_mean": float(fake_d.mean()),
            "real_delta_std": float(real_d.std()),
            "fake_delta_std": float(fake_d.std()),
            "mean_abs_gap": float(abs(real_d.mean() - fake_d.mean())),
            "std_ratio": float(fake_d.std() / (real_d.std() + 1e-9)),
        }

