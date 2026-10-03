"""Пространственно-временные энкодеры: ``[B,N,T,F] -> [B,N,H]``."""
from .base import Encoder
from .graph import DirectedHypergraph, GATLSTM, GCNGRU
from .sequence import GatedTCN, NodeTransformer

ENCODERS: dict[str, type[Encoder]] = {
    "gcn_gru": GCNGRU,
    "gat_lstm": GATLSTM,
    "hypergraph": DirectedHypergraph,
    "gated_tcn": GatedTCN,
    "transformer": NodeTransformer,
}

#: Человекочитаемые подписи для фигур и таблиц отчёта.
ENCODER_LABELS: dict[str, str] = {
    "gcn_gru": "GCN+GRU (фикс. граф)",
    "gat_lstm": "GAT+LSTM (обуч. внимание)",
    "hypergraph": "Гиперграф+Bi-GLU",
    "gated_tcn": "Gated TCN (без графа)",
    "transformer": "Transformer (без графа)",
}

__all__ = [
    "Encoder",
    "GCNGRU",
    "GATLSTM",
    "DirectedHypergraph",
    "GatedTCN",
    "NodeTransformer",
    "ENCODERS",
    "ENCODER_LABELS",
]


def build_encoder(name: str, **kwargs) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"неизвестный энкодер {name!r}; доступны: {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)
