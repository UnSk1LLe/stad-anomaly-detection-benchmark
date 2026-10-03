"""Бейзлайны и протокольные контроли."""
from .classical_aid import CaliforniaAlgorithm, StandardNormalDeviate
from .controls import (
    BaselineScorer,
    ConstantScorer,
    IsolationForestBaseline,
    PCABaseline,
    RandomScorer,
)

BASELINES: dict[str, type[BaselineScorer]] = {
    "random": RandomScorer,
    "constant": ConstantScorer,
    "pca": PCABaseline,
    "iforest": IsolationForestBaseline,
    "california": CaliforniaAlgorithm,
    "snd": StandardNormalDeviate,
}

BASELINE_LABELS: dict[str, str] = {
    "random": "Случайный score (контроль)",
    "constant": "Константа (контроль)",
    "pca": "PCA (линейная граница)",
    "iforest": "Isolation Forest",
    "california": "California алгоритм (AID)",
    "snd": "Standard Normal Deviate (AID)",
}

#: Какие из них — протокольные контроли, а не конкурирующие методы.
PROTOCOL_CONTROLS: tuple[str, ...] = ("random", "constant", "untrained")

__all__ = [
    "BaselineScorer", "RandomScorer", "ConstantScorer", "PCABaseline",
    "IsolationForestBaseline", "CaliforniaAlgorithm", "StandardNormalDeviate",
    "BASELINES", "BASELINE_LABELS", "PROTOCOL_CONTROLS",
]


def build_baseline(name: str, **kwargs) -> BaselineScorer:
    if name not in BASELINES:
        raise ValueError(f"неизвестный бейзлайн {name!r}; доступны: {sorted(BASELINES)}")
    return BASELINES[name](**kwargs)
