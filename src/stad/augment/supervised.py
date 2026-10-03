"""Supervised-арм: обучение на обоих классах под MIL-формулировку.

Почему для аугментации нужен отдельный режим обучения. Двенадцать
конфигураций основной сетки — unsupervised: они учат распределение
**нормы**, и обучающая выборка у них очищена от событий. Подмешать туда
синтетические аномалии нельзя: это загрязнит модель нормы и ухудшит
детектор. Поэтому «расширение датасета редкими аномалиями» измеряется
на отдельном арме, где обучение идёт на обоих классах.

**Формулировка — multiple-instance learning.** Метки FT-AED
корридор-уровневые: известно, что в окне было событие, но неизвестно, на
каких узлах. Это в точности постановка MIL: окно — мешок, узлы —
экземпляры, метка у мешка. Логит мешка получается max-пулингом по узлам,
и обучение идёт по нему. Та же схема используется в weakly-supervised
video anomaly detection, где метка известна на уровне ролика, а не кадра.

Энкодеры берутся **те же самые**, что в основной сетке. Благодаря этому
вопрос «какая архитектура выигрывает от аугментации» задаётся корректно:
меняется только обучающая выборка, не представление.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..encoders import build_encoder


class MILHead(nn.Module):
    """Поузловые логиты, решение мешка — максимум по узлам.

    Max-пулинг, а не среднее: инцидент затрагивает часть коридора, и
    усреднение по 196 узлам размывает сигнал до неразличимости. Это та же
    причина, по которой в основной сетке score агрегируется максимумом.
    """

    def __init__(self, hidden: int, *, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def node_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)                       # [B, N]

    def bag_logit(self, h: torch.Tensor) -> torch.Tensor:
        return self.node_logits(h).max(dim=1).values         # [B]


@dataclass
class SupervisedConfig:
    """Расписание обучения. Одинаково для всех веток эксперимента."""

    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    grad_clip: float = 1.0
    hidden: int = 96
    device: str = "cpu"
    pos_weight_cap: float = 10.0


class SupervisedDetector:
    """Энкодер основной сетки + MIL-голова, обучаемые на двух классах."""

    def __init__(self, encoder_name: str, n_features: int, n_nodes: int, window: int,
                 cfg: SupervisedConfig, *, encoder_kwargs: dict | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.encoder_name = encoder_name
        self.encoder = build_encoder(
            encoder_name, n_features=n_features, hidden=cfg.hidden,
            n_nodes=n_nodes, window=window, **(encoder_kwargs or {}),
        ).to(self.device)
        self.head = MILHead(cfg.hidden).to(self.device)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )

    def set_graph(self, A: np.ndarray) -> None:
        if hasattr(self.encoder, "set_graph"):
            self.encoder.set_graph(torch.from_numpy(A).to(self.device))

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_norm: np.ndarray,
        X_anom: np.ndarray,
        *,
        X_val_norm: np.ndarray | None = None,
        X_val_anom: np.ndarray | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> "SupervisedDetector":
        """Обучить на нормальных и аномальных окнах.

        Дисбаланс компенсируется весом положительного класса, но
        ограниченным сверху: при 20-кратном дисбалансе некапированный
        ``pos_weight`` приводит к детектору, который помечает всё подряд.
        """
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        X = np.concatenate([X_norm, X_anom]).astype(np.float32)
        y = np.concatenate([np.zeros(len(X_norm)), np.ones(len(X_anom))]).astype(np.float32)
        order = rng.permutation(len(X))
        X, y = X[order], y[order]

        pos_w = min(self.cfg.pos_weight_cap, max(1.0, len(X_norm) / max(1, len(X_anom))))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=self.device))
        params = list(self.encoder.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        has_val = X_val_norm is not None and X_val_anom is not None and len(X_val_anom) > 0
        if has_val:
            Xv = np.concatenate([X_val_norm, X_val_anom]).astype(np.float32)
            yv = np.concatenate([np.zeros(len(X_val_norm)), np.ones(len(X_val_anom))]).astype(np.float32)

        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.cfg.epochs):
            self.encoder.train(); self.head.train()
            idx = rng.permutation(len(X))
            for i in range(0, len(X), self.cfg.batch_size):
                sel = idx[i : i + self.cfg.batch_size]
                xb = torch.from_numpy(X[sel]).to(self.device)
                yb = torch.from_numpy(y[sel]).to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = crit(self.head.bag_logit(self.encoder(xb)), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                opt.step()

            if not has_val:
                continue
            self.encoder.eval(); self.head.eval()
            with torch.no_grad():
                losses = []
                for i in range(0, len(Xv), self.cfg.batch_size):
                    xb = torch.from_numpy(Xv[i : i + self.cfg.batch_size]).to(self.device)
                    yb = torch.from_numpy(yv[i : i + self.cfg.batch_size]).to(self.device)
                    losses.append(float(crit(self.head.bag_logit(self.encoder(xb)), yb)))
                v = float(np.mean(losses))
            if verbose and epoch % 5 == 0:
                print(f"    epoch {epoch:3d} val {v:.4f}")
            if v < best - 1e-5:
                best, stale = v, 0
                best_state = {
                    "e": {k: t.detach().clone() for k, t in self.encoder.state_dict().items()},
                    "h": {k: t.detach().clone() for k, t in self.head.state_dict().items()},
                }
            else:
                stale += 1
                if stale >= self.cfg.patience:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["e"])
            self.head.load_state_dict(best_state["h"])
        self.best_val_loss = best
        return self

    # ---------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, X: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
        """``-> [n, N]`` поузловые логиты; агрегация до коридора — в метриках."""
        self.encoder.eval(); self.head.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(self.device)
            out.append(self.head.node_logits(self.encoder(xb)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
