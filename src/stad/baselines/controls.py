"""Протокольные контроли. Без них таблица сравнения не защищается.

Это не «слабые модели для контраста», а **инструменты валидации
протокола**. Каждый проверяет конкретное утверждение:

:class:`RandomScorer`
    Проверяет, что метрика не сломана. Kim et al. (2021) показали, что
    при point-adjustment случайный score превращается в SOTA. Если в
    наших результатах случайный контроль попадает в верхнюю половину по
    основной метрике — метрика выбрана неверно, и все остальные числа
    недействительны. Ожидаемое поведение: ``average_precision ≈
    prevalence``, ``padf ≈ 0``.

:class:`UntrainedModel`
    Проверяет, что обучение вообще что-то даёт. У Kim et al. необученная
    модель оказалась сравнима с опубликованными методами даже без PA.
    Разность «обученная − необученная» и есть измеренный вклад обучения;
    если он мал, архитектурные различия обсуждать бессмысленно.

:class:`PCABaseline`
    Линейная нижняя граница. Sehili et al. (2023): PCA превосходит
    многие недавние DL-подходы на популярных бенчмарках; Alves et al.
    (2026) независимо подтвердили на SMD (100 прогонов × 28 машин), что
    PCA не хуже OmniAnomaly, когда point-adjustment не применяется.
    **Если ST-GNN не отрывается от PCA значимо — это и есть результат
    диссертации, а не неудача эксперимента.**
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..data.types import SplitData


def calib_as_test(data: SplitData) -> SplitData:
    """Копия ``data``, где «тестом» служат калибровочные окна без меток.

    Бейзлайны читают только ``X_test`` и форму ``y_test``, поэтому так они скорят
    калибровочную выборку тем же кодом, что и тест: порог калибруется на
    ней, тестовые метки в калибровке не участвуют.
    """
    if data.X_calib is None or data.t_calib is None:
        raise ValueError("у данных нет калибровочной выборки (X_calib/t_calib)")
    n = len(data.X_calib)
    return dataclasses.replace(
        data,
        X_test=data.X_calib,
        y_test=np.zeros((n, data.n_nodes), dtype=np.int64),
        event_id_test=np.full((n, data.n_nodes), -1, dtype=np.int64),
        t_test=data.t_calib,
    )


class BaselineScorer:
    """Единый интерфейс небучаемых и слабообучаемых бейзлайнов."""

    name = "baseline"
    n_params = 0

    def fit(self, data: SplitData) -> "BaselineScorer":  # noqa: D401
        return self

    def score(self, data: SplitData) -> np.ndarray:
        """``-> [n_test, N]``, больше = аномальнее."""
        raise NotImplementedError

    def score_calib(self, data: SplitData) -> np.ndarray:
        """Score калибровочных окон ``[n_calib, N]`` — для калибровки порога."""
        return self.score(calib_as_test(data))


class RandomScorer(BaselineScorer):
    """Случайный score. Детектор без навыка — контроль метрики."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def score(self, data: SplitData) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal(data.y_test.shape).astype(np.float32)

    def score_calib(self, data: SplitData) -> np.ndarray:
        # независимый поток: иначе score калибровки были бы началом тестового ряда
        rng = np.random.default_rng((self.seed, 1))
        return rng.standard_normal((len(data.X_calib), data.n_nodes)).astype(np.float32)


class ConstantScorer(BaselineScorer):
    """Константа. Крайний случай: любая метрика должна дать минимум.

    Полезен как проверка кода метрик: AP должен равняться prevalence,
    event_recall при FPR 1% — быть близким к нулю.
    """

    name = "constant"

    def score(self, data: SplitData) -> np.ndarray:
        return np.zeros(data.y_test.shape, dtype=np.float32)


class PCABaseline(BaselineScorer):
    """Ошибка реконструкции линейного PCA, обученного на нормальных окнах.

    Окно разворачивается в вектор ``N*T*F``; компоненты выбираются по
    доле объяснённой дисперсии. Это честный конкурент: у него нет
    пространственного prior, но есть вся информация окна.
    """

    name = "pca"

    def __init__(self, variance: float = 0.95, max_components: int = 128) -> None:
        self.variance = variance
        self.max_components = max_components
        self._pca = None
        self._shape: tuple[int, ...] | None = None

    def fit(self, data: SplitData) -> "PCABaseline":
        from sklearn.decomposition import PCA

        X = data.X_train
        flat = X.reshape(len(X), -1)
        n_comp = min(self.max_components, flat.shape[0] - 1, flat.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0).fit(flat)
        cum = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cum, self.variance) + 1)
        self._pca = PCA(n_components=min(keep, n_comp), svd_solver="randomized", random_state=0).fit(flat)
        self._shape = X.shape[1:]
        self.n_params = int(self._pca.components_.size)
        return self

    def score(self, data: SplitData) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCABaseline не обучен")
        X = data.X_test
        flat = X.reshape(len(X), -1)
        rec = self._pca.inverse_transform(self._pca.transform(flat)).reshape(X.shape)
        # ошибка агрегируется по времени и признакам, оставляя ось узлов
        return ((X - rec) ** 2).mean(axis=(2, 3)).astype(np.float32)


class IsolationForestBaseline(BaselineScorer):
    """Adaptive-Isolation-Forest-подобный бейзлайн на пер-узловых окнах.

    Мотивация: в работе «Real-time anomaly detection of short-term traffic
    disruptions in urban areas through adaptive isolation forest» (2025)
    неглубокий метод даёт detection rate 71–100% на **реальных** городских
    событиях — выше, чем можно ожидать от «простого» бейзлайна. Он включён
    именно поэтому, а не для контраста.

    Каждый узел скорится отдельно (признаки окна как вектор), что
    соответствует постановке «локальный детектор на каждом детекторе».
    """

    name = "iforest"

    def __init__(self, n_estimators: int = 150, max_samples: int = 512, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self._model = None

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        """``[n,N,T,F] -> [n*N, 4F]``: среднее, стд, последнее, приращение."""
        mean = X.mean(2)
        std = X.std(2)
        last = X[:, :, -1, :]
        delta = X[:, :, -1, :] - X[:, :, 0, :]
        feat = np.concatenate([mean, std, last, delta], axis=-1)
        return feat.reshape(-1, feat.shape[-1])

    def fit(self, data: SplitData) -> "IsolationForestBaseline":
        from sklearn.ensemble import IsolationForest

        F = self._features(data.X_train)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(F)),
            random_state=self.seed,
            contamination="auto",
        ).fit(F)
        self.n_params = int(self.n_estimators)
        return self

    def score(self, data: SplitData) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsolationForestBaseline не обучен")
        F = self._features(data.X_test)
        # score_samples: больше = нормальнее, поэтому инвертируем
        s = -self._model.score_samples(F)
        return s.reshape(data.y_test.shape).astype(np.float32)
