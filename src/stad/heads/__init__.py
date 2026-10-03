"""Головы детекции: ``[B,N,H] (+ x) -> score [B,N]``."""
from .adversarial import BiGANHead
from .base import Head
from .contrastive import ContrastiveHead
from .density import FlowHead
from .forecast import ForecastHead, PhysicsResidualHead
from .reconstruction import ReconHead

HEADS: dict[str, type[Head]] = {
    "recon": ReconHead,
    "bigan": BiGANHead,
    "flow": FlowHead,
    "forecast": ForecastHead,
    "contrastive": ContrastiveHead,
    "physics": PhysicsResidualHead,
}

#: Подписи для фигур.
HEAD_LABELS: dict[str, str] = {
    "recon": "Реконструкция",
    "bigan": "Критик BiGAN",
    "flow": "Плотность (flow)",
    "forecast": "Остаток прогноза",
    "contrastive": "Контрастив",
    "physics": "Невязка LWR",
}

#: Механизм score — для группировки в отчёте.
HEAD_MECHANISM: dict[str, str] = {
    "recon": "reconstruction",
    "bigan": "discriminator",
    "flow": "density",
    "forecast": "forecast-residual",
    "contrastive": "representation-discrepancy",
    "physics": "physical-residual",
}

__all__ = [
    "Head",
    "ReconHead",
    "BiGANHead",
    "FlowHead",
    "ForecastHead",
    "PhysicsResidualHead",
    "ContrastiveHead",
    "HEADS",
    "HEAD_LABELS",
    "HEAD_MECHANISM",
]


def build_head(name: str, **kwargs) -> Head:
    if name not in HEADS:
        raise ValueError(f"неизвестная голова {name!r}; доступны: {sorted(HEADS)}")
    return HEADS[name](**kwargs)
