"""Тесты архитектур: формы, отсутствие утечки цели, выравнивание бюджета."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from stad.budget import match_budget
from stad.encoders import ENCODERS, build_encoder
from stad.heads import HEADS, build_head
from stad.model import build_detector

B, N, T, F, H = 3, 9, 12, 3, 32


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(B, N, T, F)


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoder_output_shape(name, batch):
    """Контракт ``[B,N,T,F] -> [B,N,H]`` обязателен для всех энкодеров."""
    enc = build_encoder(name, n_features=F, hidden=H, n_nodes=N, window=T)
    if hasattr(enc, "set_graph"):
        enc.set_graph(torch.eye(N))
    out = enc(batch)
    assert out.shape == (B, N, H), f"{name}: {tuple(out.shape)}"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_encoder_rejects_wrong_shape(name):
    """Неверная форма должна падать явно, а не тихо ломать результаты."""
    enc = build_encoder(name, n_features=F, hidden=H, n_nodes=N, window=T)
    with pytest.raises(ValueError):
        enc(torch.randn(B, N + 1, T, F))


@pytest.mark.parametrize("head_name", sorted(HEADS))
def test_head_score_shape_and_grad(head_name):
    """Голова даёт ``[B,N]`` и пропускает градиент в энкодер."""
    det = build_detector(
        "gcn_gru", head_name, hidden=H, n_features=F, n_nodes=N, window=T,
        head_kwargs={"lanes": 3} if head_name == "physics" else {},
    )
    det.set_graph(torch.eye(N))
    x = torch.randn(B, N, T, F)

    loss = det.loss(x)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in det.encoder.parameters() if p.grad is not None]
    assert grads, f"{head_name}: градиент не доходит до энкодера"

    s = det.score(x)
    assert s.shape == (B, N), f"{head_name}: score {tuple(s.shape)}"
    assert torch.isfinite(s).all()


def test_forecast_head_does_not_leak_target():
    """Критично: энкодер не должен видеть целевые шаги прогноза.

    Проверка — изменение только последних ``horizon`` шагов окна не
    должно менять представление энкодера. Если меняет, значит цель
    утекает, и качество завышено.
    """
    det = build_detector("gcn_gru", "forecast", hidden=H, n_features=F,
                         n_nodes=N, window=T, head_kwargs={"horizon": 3})
    det.set_graph(torch.eye(N))
    det.eval()
    x = torch.randn(B, N, T, F)
    x2 = x.clone()
    x2[:, :, -3:, :] += 100.0

    with torch.no_grad():
        h1, h2 = det.encode(x), det.encode(x2)
    assert torch.allclose(h1, h2, atol=1e-6), "цель прогноза утекает в энкодер"

    with torch.no_grad():
        s1, s2 = det.score(x), det.score(x2)
    assert not torch.allclose(s1, s2), "score не реагирует на изменение цели"


def test_bigan_discriminator_excluded_from_main_params():
    """Критик обучается отдельным оптимизатором, иначе он не состязательный."""
    det = build_detector("gcn_gru", "bigan", hidden=H, n_features=F, n_nodes=N, window=T)
    main = {id(p) for p in det.main_parameters()}
    d_params = {id(p) for p in det.head.d_parameters()}
    assert d_params, "критик без параметров"
    assert not (main & d_params), "параметры критика попали в основной оптимизатор"


def test_bigan_score_components():
    """Абляция «цикл против критика» должна быть доступна явно."""
    det = build_detector("gcn_gru", "bigan", hidden=H, n_features=F, n_nodes=N, window=T)
    det.set_graph(torch.eye(N))
    h = det.encode(torch.randn(B, N, T, F))
    parts = det.head.score_components(h)
    assert set(parts) == {"cycle", "critic", "combined"}
    for v in parts.values():
        assert v.shape == (B, N)


def test_flow_head_gives_higher_score_to_outliers():
    """Плотностная голова: у выброса ``-log p`` должен быть выше.

    Проверка семантики, а не кода: если знак перепутан, вся таблица
    инвертируется, и это не поймается тестами на формы.
    """
    torch.manual_seed(0)
    head = build_head("flow", hidden=16, n_features=F, window=T, n_nodes=N)
    normal = torch.randn(256, 16) * 0.5
    opt = torch.optim.Adam(head.parameters(), lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = head.loss(normal)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_norm = head.score(torch.randn(128, 16) * 0.5).mean()
        s_out = head.score(torch.randn(128, 16) * 0.5 + 8.0).mean()
    assert s_out > s_norm, f"выброс получил меньший score: {s_out:.2f} vs {s_norm:.2f}"


@pytest.mark.parametrize("encoder", ["gcn_gru", "gat_lstm", "hypergraph", "transformer", "gated_tcn"])
def test_budget_matching_converges(encoder):
    """Бюджет параметров выравнивается — условие корректности сравнения."""
    br = match_budget(
        encoder, "recon", target=120_000,
        n_features=F, n_nodes=N, window=T, tolerance=0.25,
        head_kwargs={}, encoder_kwargs={},
    )
    assert br.hidden > 0
    assert br.within_tolerance, (
        f"{encoder}: подобрано {br.n_params} при цели 120000 "
        f"(отклонение {br.deviation:.1%}, hidden={br.hidden})"
    )


def test_budget_matched_configs_are_comparable():
    """Две архитектуры при одной цели должны иметь близкое число параметров."""
    targets = {}
    for enc in ("gcn_gru", "transformer"):
        br = match_budget(enc, "recon", target=120_000, n_features=F, n_nodes=N,
                          window=T, tolerance=0.3)
        targets[enc] = br.n_params
    a, b = targets.values()
    assert abs(a - b) / max(a, b) < 0.45, f"бюджеты несопоставимы: {targets}"


# ------------------------------------------------------------- чекпойнты
def test_checkpoint_roundtrip_preserves_scores(tmp_path):
    """Загруженная модель обязана давать те же score, что сохранённая.

    Это главное свойство чекпойнта. Рецепт сборки (hidden подбирается
    под бюджет параметров), матрица смежности и веса должны
    восстанавливаться вместе: подстановка другого графа тихо изменила бы
    поведение модели, обученной под конкретную структуру.
    """
    import numpy as np

    from stad.checkpoints import describe, load_detector, save_detector

    A = np.eye(N, dtype=np.float32)
    det = build_detector("gat_lstm", "flow", hidden=H, n_features=F, n_nodes=N, window=T)
    det.set_graph(torch.from_numpy(A))
    det.eval()
    x = torch.randn(2, N, T, F)
    before = det.score(x)

    path = tmp_path / "ckpt.pt"
    save_detector(det, path, encoder="gat_lstm", head="flow", hidden=H,
                  n_features=F, n_nodes=N, window=T, adjacency=A,
                  extra={"config": "test", "seed": 0})

    restored, meta = load_detector(path)
    after = restored.score(x)

    assert torch.allclose(before, after, atol=1e-6), "score после загрузки изменился"
    assert meta["recipe"]["encoder"] == "gat_lstm"
    assert meta["recipe"]["hidden"] == H
    assert meta["extra"]["config"] == "test"
    assert describe(path)["n_params"] == det.n_params


def test_checkpoint_rejects_wrong_format_version(tmp_path):
    """Чекпойнт чужой версии должен падать явно, а не собирать мусор."""
    from stad.checkpoints import load_detector

    path = tmp_path / "old.pt"
    torch.save({"format_version": 0, "state_dict": {}, "recipe": {}}, path)
    with pytest.raises(ValueError, match="версия формата"):
        load_detector(path)
