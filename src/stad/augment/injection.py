"""Инъекция аномалий: интерфейс и generator-independent вариант на физике.

Зачем вообще нужен не-GAN инжектор. В диссертации зафиксирован пробел
«циркулярность оценки»: если аномалии **генерируются** и **детектируются**
моделями одного семейства, детектор учится узнавать артефакты генератора,
а не аномалии. Тогда прирост от аугментации — артефакт эксперимента.

Защита строится из двух частей, и обе обязательны:

1. **Архитектурная несхожесть.** Аугментирующий GAN (``augment/gan.py``)
   сделан свёрточным WGAN-GP на остатках, а не BiGAN на представлениях,
   как детекторная голова. Общих компонентов у них нет.
2. **Физический контроль.** Этот модуль инжектирует аномалию из закона
   сохранения LWR — без единого обученного параметра. Если прирост
   качества одинаков при GAN-инъекции и при физической, значит помогает
   сам факт расширения редкого класса, а не выученное распределение.
   Если прирост есть только у GAN — нужно доказывать, что это не
   запоминание артефактов.

Оценка в обоих случаях ведётся **только на реальных событиях**.
Синтетика живёт исключительно в обучающей выборке.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Injector(ABC):
    """Превращает нормальное окно в аномальное.

    Контракт: ``[B, N, T, F] -> [B, N, T, F]`` плюс маска
    затронутых узлов ``[B, N]``. Исходное окно — реальное, меняется
    только сигнатура аномалии: так сохраняется настоящий контекст
    (суточный профиль, узкое место, шум детекторов), а синтетической
    остаётся только та часть, которой в данных не хватает.
    """

    name = "injector"
    needs_fit = False

    def fit(self, X_normal: np.ndarray, X_anomalous: np.ndarray | None = None) -> "Injector":
        return self

    @abstractmethod
    def inject(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``-> (окна с аномалией, маска затронутых узлов)``."""


class LWRShockInjector(Injector):
    """Ударная волна по закону сохранения LWR. Обученных параметров нет.

    Физика. Инцидент на посту *m* создаёт волну снижения скорости,
    распространяющуюся **вверх** по потоку со скоростью порядка
    15–20 км/ч; занятость растёт, поток падает. Ниже по потоку эффект
    обратный и слабый. Та же модель используется в синтетическом
    коридоре (``data/synthetic.py``), что делает контроль согласованным.

    Параметры возмущения берутся из диапазонов, а не из данных, поэтому
    инжектор не зависит ни от какой обученной модели — в этом его смысл.

    Возмущение **аддитивное, в единицах sigma**: данные приходят
    z-нормализованными, и мультипликативное масштабирование там меняло бы
    знак эффекта в зависимости от знака значения. Амплитуда 1–2.5 sigma
    соответствует падению скорости, наблюдаемому при реальном ДТП.
    """

    name = "lwr"

    def __init__(
        self,
        *,
        n_lanes: int = 4,
        speed_idx: int = 0,
        occ_idx: int = 1,
        vol_idx: int = 2,
        amplitude: tuple[float, float] = (1.0, 2.5),      # в единицах sigma
        spread_stations: tuple[float, float] = (1.5, 5.0),
        onset_frac: tuple[float, float] = (0.2, 0.7),
    ) -> None:
        self.n_lanes = n_lanes
        self.speed_idx = speed_idx
        self.occ_idx = occ_idx
        self.vol_idx = vol_idx
        self.amplitude = amplitude
        self.spread_stations = spread_stations
        self.onset_frac = onset_frac

    def inject(self, X: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        B, N, T, F = X.shape
        L = self.n_lanes
        M = N // L
        out = X.copy()
        affected = np.zeros((B, N), dtype=bool)

        station = np.arange(N) // L                       # индекс поста для каждого узла

        for b in range(B):
            m0 = int(rng.integers(1, max(2, M)))          # эпицентр
            amp = float(rng.uniform(*self.amplitude))
            spread = float(rng.uniform(*self.spread_stations))
            t0 = int(T * float(rng.uniform(*self.onset_frac)))
            lane_hit = rng.random(L) < 0.75                # часть полос может уцелеть
            if not lane_hit.any():
                lane_hit[int(rng.integers(0, L))] = True

            for t in range(t0, T):
                # фронт движется вверх по потоку: эпицентр смещается к меньшим постам
                front = m0 - 0.35 * (t - t0)
                d = station - front
                spatial = np.exp(-0.5 * (np.clip(d, None, 0.0) / spread) ** 2)
                spatial = np.where(d > 0.5, 0.1 * spatial, spatial)   # ниже по потоку слабо
                spatial = spatial * np.tile(lane_hit, M)
                ramp = min(1.0, (t - t0 + 1) / max(1.0, 0.25 * (T - t0)))
                k = amp * ramp * spatial                              # [N]

                # знаки из физики: скорость падает, занятость растёт,
                # поток падает (пропускная способность снижается)
                out[b, :, t, self.speed_idx] -= k
                out[b, :, t, self.occ_idx] += 1.6 * k
                out[b, :, t, self.vol_idx] -= 0.5 * k
                affected[b] |= k > 0.2

        return out.astype(np.float32), affected
