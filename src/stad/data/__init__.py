"""Слой данных: единый контракт для всех архитектур."""
from .ft_aed import DataNotDownloaded, corridor_adjacency, load_ft_aed
from .labels import align_incidents, windows_from_panel
from .splits import Standardizer, build_split, drop_anomalous_windows, time_split_indices
from .synthetic import make_synthetic_corridor
from .types import SplitData

__all__ = [
    "SplitData",
    "align_incidents",
    "windows_from_panel",
    "build_split",
    "drop_anomalous_windows",
    "time_split_indices",
    "Standardizer",
    "make_synthetic_corridor",
    "load_ft_aed",
    "corridor_adjacency",
    "DataNotDownloaded",
]


def load_dataset(name: str, **kwargs) -> SplitData:
    """Фабрика датасетов по имени из конфига."""
    if name in {"synthetic", "synthetic-corridor"}:
        return make_synthetic_corridor(**kwargs)
    if name in {"ft-aed", "ft_aed", "FT-AED"}:
        return load_ft_aed(**kwargs)
    raise ValueError(f"неизвестный датасет: {name!r}; доступны: synthetic, ft-aed")
