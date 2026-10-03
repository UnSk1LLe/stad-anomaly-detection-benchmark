"""Классические алгоритмы автоматического обнаружения инцидентов (AID).

Почему они обязательны в диссертации по транспорту. Вся современная
литература по детекции аномалий трафика — это ML/DL, и в такой подборке
легко упустить доменную линию, которая существует с 1970-х и до сих
пор эксплуатируется в реальных системах управления магистралями.
Рецензент транспортного Q1-журнала спросит, почему новая архитектура не
сравнена с каноном. Ответ «мы сравнили с графовыми автоэнкодерами» его
не устроит.

Реализованы два канонических семейства:

:class:`CaliforniaAlgorithm`
    Алгоритм «California» (семейство, развитое из работ Payne и Tignor,
    1970-е). Решение принимается по **пространственным разностям
    занятости** между соседними постами: инцидент создаёт перепад —
    выше по потоку занятость растёт, ниже падает. Три порога и логика
    подтверждения по последовательным интервалам.

:class:`StandardNormalDeviate`
    SND: отклонение текущего значения от скользящего среднего в единицах
    скользящего стандартного отклонения. По сути — классический
    статистический контроль процесса, применённый к детектору.

Оба метода **не обучаются на данных** в современном смысле: их параметры
калибруются по нормальной части обучающей выборки, что честно и
соответствует их исходному применению.
"""
from __future__ import annotations

import numpy as np

from ..data.types import SplitData
from .controls import BaselineScorer


class CaliforniaAlgorithm(BaselineScorer):
    """California-подобный детектор по пространственным разностям занятости.

    Для узла *i* на полосе *l* и его соседа ниже по потоку *i+lanes*:

    * ``OCCDF``  = occ(i) − occ(i+L)                     — абсолютная разность
    * ``OCCRDF`` = (occ(i) − occ(i+L)) / occ(i)          — относительная
    * ``DOCCTD`` = occ(i+L, t) − occ(i+L, t−k)           — падение ниже по потоку

    Скор формируется как мягкая комбинация превышений трёх порогов —
    так метод вписывается в общий протокол (ранжирование по непрерывному
    score и порог по FPR), оставаясь при этом тем же алгоритмом.
    """

    name = "california"

    def __init__(
        self,
        *,
        occ_idx: int = 1,
        lanes: int = 3,
        lag: int = 2,
        calibrate_quantile: float = 0.995,
    ) -> None:
        self.occ_idx = occ_idx
        self.lanes = lanes
        self.lag = lag
        self.calibrate_quantile = calibrate_quantile
        self.thresholds: dict[str, float] = {}

    # ------------------------------------------------------------------ внутр.
    def _components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occ = X[..., self.occ_idx]                            # [n, N, T]
        L = self.lanes
        n, N, T = occ.shape

        down = np.empty_like(occ)
        if N > L:
            down[:, : N - L] = occ[:, L:]
            down[:, N - L :] = occ[:, N - L :]                # край: сам себя
        else:
            down[:] = occ

        now, dn = occ[..., -1], down[..., -1]
        occdf = now - dn
        with np.errstate(divide="ignore", invalid="ignore"):
            occrdf = np.where(now > 1e-6, occdf / now, 0.0)
        k = min(self.lag, T - 1)
        docctd = down[..., -1 - k] - dn                       # падение занятости ниже по потоку
        return occdf, occrdf, docctd

    def fit(self, data: SplitData) -> "CaliforniaAlgorithm":
        """Калибровка порогов по верхнему квантилю нормальных окон."""
        occdf, occrdf, docctd = self._components(data.X_train)
        q = self.calibrate_quantile
        self.thresholds = {
            "occdf": float(np.quantile(occdf, q)),
            "occrdf": float(np.quantile(occrdf, q)),
            "docctd": float(np.quantile(docctd, q)),
        }
        self.n_params = 3
        return self

    def score(self, data: SplitData) -> np.ndarray:
        if not self.thresholds:
            raise RuntimeError("CaliforniaAlgorithm не калиброван")
        occdf, occrdf, docctd = self._components(data.X_test)
        t = self.thresholds

        def soft(v: np.ndarray, thr: float) -> np.ndarray:
            scale = abs(thr) + 1e-6
            return np.clip((v - thr) / scale, -1.0, None)

        s = soft(occdf, t["occdf"]) + soft(occrdf, t["occrdf"]) + soft(docctd, t["docctd"])
        return (s / 3.0).astype(np.float32)


class StandardNormalDeviate(BaselineScorer):
    """SND: отклонение в единицах скользящего стандартного отклонения.

    Классика статистического контроля: ``z = (x_t − μ_window) / σ_window``.
    Считается по каждому признаку внутри окна и агрегируется максимумом —
    инцидент проявляется хотя бы в одном канале, и усреднение его размывает.
    """

    name = "snd"

    def __init__(self, *, min_sigma: float = 1e-3) -> None:
        self.min_sigma = min_sigma
        self.feature_weights: np.ndarray | None = None

    def fit(self, data: SplitData) -> "StandardNormalDeviate":
        """Вес признака обратен его типичному |z| на норме.

        Без этого шага самый шумный канал доминирует в максимуме и
        метод вырождается в детектор шума.
        """
        z = self._z(data.X_train)
        typical = np.quantile(np.abs(z), 0.99, axis=(0, 1))
        self.feature_weights = (1.0 / np.maximum(typical, 1e-6)).astype(np.float32)
        self.n_params = int(self.feature_weights.size)
        return self

    def _z(self, X: np.ndarray) -> np.ndarray:
        """``[n,N,T,F] -> [n,N,F]`` z-отклонение последнего шага окна."""
        hist = X[:, :, :-1, :]
        mu = hist.mean(axis=2)
        sd = np.maximum(hist.std(axis=2), self.min_sigma)
        return (X[:, :, -1, :] - mu) / sd

    def score(self, data: SplitData) -> np.ndarray:
        z = np.abs(self._z(data.X_test))
        if self.feature_weights is not None:
            z = z * self.feature_weights[None, None, :]
        return z.max(axis=-1).astype(np.float32)
