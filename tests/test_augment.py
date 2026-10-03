"""Тесты аугментации редкого класса.

Проверяется не «запускается ли код», а свойства, без которых эксперимент
не имеет смысла: инъекция должна быть физически осмысленной, GAN —
аддитивным в единицах sigma, MIL-голова — агрегировать мешок максимумом,
а генератор не должен зависеть от детектора.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from stad.augment import (
    AnomalyGAN,
    GANConfig,
    LWRShockInjector,
    MILHead,
    SupervisedConfig,
    SupervisedDetector,
)

B, N, T, F, LANES = 6, 24, 8, 3, 4
SPEED, OCC, VOL = 0, 1, 2


@pytest.fixture
def normal():
    """Нормальные окна в z-нормализованной шкале, как их даёт загрузчик."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((B, N, T, F)).astype(np.float32)


# ------------------------------------------------------------------ физика
def test_lwr_injection_has_correct_signs(normal):
    """Скорость падает, занятость растёт, поток падает — иначе это не ДТП."""
    inj = LWRShockInjector(n_lanes=LANES, amplitude=(2.0, 2.0))
    out, affected = inj.inject(normal, np.random.default_rng(0))

    d = out - normal
    hit = affected
    assert hit.any(), "ни один узел не затронут"
    assert d[..., SPEED][hit].mean() < 0, "скорость должна падать"
    assert d[..., OCC][hit].mean() > 0, "занятость должна расти"
    assert d[..., VOL][hit].mean() < 0, "поток должен падать"


def test_lwr_injection_is_spatially_local(normal):
    """Инцидент затрагивает часть коридора, а не всю сеть сразу."""
    inj = LWRShockInjector(n_lanes=LANES)
    _, affected = inj.inject(normal, np.random.default_rng(1))
    share = affected.mean(axis=1)
    assert share.max() < 0.95, f"затронуто {share.max():.0%} узлов — это не локальный инцидент"
    assert share.mean() > 0.02, "возмущение слишком слабое, чтобы быть аномалией"


def test_lwr_injection_appears_partway_through_window(normal):
    """Аномалия начинается внутри окна, а не присутствует с первого шага.

    Это важно для детекции: окно должно содержать и норму, и переход,
    иначе модель не сможет опереться на контраст. Сам фронт нарастает
    быстро (ДТП — резкое событие), поэтому проверяется не монотонность
    по каждому шагу, а что вторая половина окна возмущена сильнее первой.
    """
    inj = LWRShockInjector(n_lanes=LANES, amplitude=(2.0, 2.0), onset_frac=(0.4, 0.6))
    out, _ = inj.inject(normal, np.random.default_rng(2))
    d = np.abs(out - normal)[..., SPEED].mean(axis=(0, 1))
    half = len(d) // 2
    assert d[half:].mean() > d[:half].mean() * 2, f"профиль по времени: {d.round(3)}"
    assert d[0] == pytest.approx(0.0, abs=1e-6), "возмущение не должно начинаться с первого шага"


def test_lwr_has_no_learned_parameters():
    """Generator-independent контроль не должен ничему обучаться.

    Если у инжектора появятся обучаемые веса, он перестанет быть защитой
    от циркулярности оценки — весь смысл этой ветви эксперимента.
    """
    inj = LWRShockInjector(n_lanes=LANES)
    assert not isinstance(inj, torch.nn.Module)
    assert not hasattr(inj, "parameters")
    assert inj.needs_fit is False


# --------------------------------------------------------------------- GAN
def test_gan_requires_enough_real_anomalies(normal):
    """На нескольких окнах GAN обучать нельзя — должно падать явно."""
    gan = AnomalyGAN(N, T, F, GANConfig(epochs=1))
    with pytest.raises(ValueError, match="аномальных окон"):
        gan.fit(normal, normal[:4])


def test_gan_injection_is_additive_and_bounded(normal):
    """Остаток аддитивный и ограничен ``max_delta`` в единицах sigma."""
    rng = np.random.default_rng(0)
    anom = normal.copy()
    anom[..., SPEED] -= 2.0
    anom = np.repeat(anom, 2, axis=0)

    cfg = GANConfig(epochs=3, batch_size=8, max_delta=1.5)
    gan = AnomalyGAN(N, T, F, cfg).fit(normal, anom, seed=0)
    out, affected = gan.inject(normal, rng)

    assert out.shape == normal.shape
    delta = out - normal
    assert np.abs(delta).max() <= cfg.max_delta + 1e-4, "остаток вышел за max_delta"
    assert affected.shape == (B, N)


def test_gan_is_architecturally_distinct_from_detector_head():
    """Аугментирующий GAN не должен делить компоненты с головой детектора.

    Иначе прирост от аугментации неотличим от запоминания артефактов
    собственного генератора — пробел «циркулярность оценки».
    """
    from stad.heads import BiGANHead

    gan = AnomalyGAN(N, T, F, GANConfig(epochs=1))
    head = BiGANHead(hidden=32, n_features=F, window=T, n_nodes=N)

    gan_types = {type(m).__name__ for m in gan.G.modules()} | {type(m).__name__ for m in gan.D.modules()}
    assert "Conv1d" in gan_types or "Conv2d" in gan_types, "GAN должен быть свёрточным"
    head_types = {type(m).__name__ for m in head.modules()}
    assert not ({"Conv1d", "Conv2d"} & head_types), "голова детектора должна быть полносвязной"


# --------------------------------------------------------------------- MIL
def test_mil_head_pools_bag_by_max():
    """Логит мешка = максимум по узлам: инцидент локален."""
    head = MILHead(hidden=16).eval()      # dropout выключен: сравниваем два прохода
    h = torch.randn(4, N, 16)
    node = head.node_logits(h)
    bag = head.bag_logit(h)
    assert node.shape == (4, N)
    assert torch.allclose(bag, node.max(dim=1).values)


def test_supervised_detector_learns_injected_signal(normal):
    """Санити: на явно разделимой задаче MIL-детектор обязан обучиться.

    Если этот тест падает, обсуждать эффект аугментации бессмысленно —
    сломан сам supervised-арм.
    """
    rng = np.random.default_rng(0)
    X_norm = rng.standard_normal((120, N, T, F)).astype(np.float32)
    X_anom, _ = LWRShockInjector(n_lanes=LANES, amplitude=(3.0, 3.0)).inject(
        rng.standard_normal((60, N, T, F)).astype(np.float32), rng
    )

    det = SupervisedDetector("gcn_gru", F, N, T, SupervisedConfig(epochs=12, hidden=32))
    det.set_graph(np.eye(N, dtype=np.float32))
    det.fit(X_norm[:100], X_anom[:50], X_val_norm=X_norm[100:], X_val_anom=X_anom[50:], seed=0)

    s_norm = det.score(X_norm[100:]).max(axis=1).mean()
    s_anom = det.score(X_anom[50:]).max(axis=1).mean()
    assert s_anom > s_norm, f"аномалии получили меньший score: {s_anom:.3f} vs {s_norm:.3f}"


def test_summarise_computes_paired_delta():
    """Эффект считается парно к базе того же блока и энкодера."""
    import pandas as pd

    from stad.augment import summarise

    df = pd.DataFrame([
        {"block": "f0|s0", "encoder": "gcn_gru", "source": "none", "ratio": 0.0, "padf": 0.20},
        {"block": "f0|s0", "encoder": "gcn_gru", "source": "gan", "ratio": 1.0, "padf": 0.30},
        {"block": "f0|s0", "encoder": "gcn_gru", "source": "lwr", "ratio": 1.0, "padf": 0.25},
    ])
    out = summarise(df)
    gan = out[out["source"] == "gan"].iloc[0]
    lwr = out[out["source"] == "lwr"].iloc[0]
    assert gan["delta_mean"] == pytest.approx(0.10)
    assert lwr["delta_mean"] == pytest.approx(0.05)
