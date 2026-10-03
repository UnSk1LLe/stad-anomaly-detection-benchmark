"""Аугментация редкого класса и факторный эксперимент по её эффекту.

Отдельный пакет, а не расширение основной сетки, потому что режим
обучения принципиально другой: двенадцать конфигураций сетки
unsupervised и учат норму, а аугментация редкого класса имеет смысл
только при обучении на обоих классах.

Состав:

``injection``
    Интерфейс инжектора и generator-independent вариант на законе
    сохранения LWR — защита от циркулярности оценки.
``gan``
    WGAN-GP на мультипликативных остатках: генерируется сигнатура
    аномалии, а контекст берётся из реальных нормальных окон.
``supervised``
    MIL-обучение тех же энкодеров на двух классах: метки
    корридор-уровневые, поэтому окно — мешок, узлы — экземпляры.
``study``
    Факторный прогон encoder × source × ratio и сводка эффекта.
"""
from .gan import AnomalyGAN, GANConfig
from .injection import Injector, LWRShockInjector
from .study import StudyConfig, run_study, summarise
from .supervised import MILHead, SupervisedConfig, SupervisedDetector

__all__ = [
    "Injector", "LWRShockInjector",
    "AnomalyGAN", "GANConfig",
    "MILHead", "SupervisedConfig", "SupervisedDetector",
    "StudyConfig", "run_study", "summarise",
]
