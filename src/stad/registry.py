"""Реестр конфигураций: что именно сравнивается и зачем.

Каждая запись несёт поле ``rationale`` — почему она в сетке. Это не
документация ради документации: при 12–27 конфигурациях легко потерять
смысл эксперимента, и тогда результат превращается в таблицу без вывода.
Поле попадает в итоговый отчёт.

Две сетки:

``core`` (12 конфигураций)
    Защитимая таблица для главы 2 диссертации: 3 протокольных контроля,
    3 доменных и неглубоких бейзлайна, 1 глубокий неграфовый и 5
    кандидатов. Это тот минимум, который закрывает вопросы рецензента.

``extended`` (15 конфигураций = 5 энкодеров × 3 головы)
    Полный крест для ответа на главный вопрос бенчмарка: **что важнее,
    энкодер или механизм score**. Нужен для фигуры-теплокарты и для
    разложения дисперсии. Включается отдельно, потому что стоит дороже.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Group = Literal["control", "baseline", "non_graph", "candidate"]


@dataclass(frozen=True)
class Config:
    """Одна строка итоговой таблицы."""

    name: str
    group: Group
    rationale: str
    #: Для обучаемых: имена энкодера и головы.
    encoder: str | None = None
    head: str | None = None
    #: Для небучаемых: имя бейзлайна из ``stad.baselines``.
    baseline: str | None = None
    #: Пропустить обучение (контроль «untrained»).
    randomize_only: bool = False
    encoder_kwargs: dict = field(default_factory=dict)
    head_kwargs: dict = field(default_factory=dict)
    baseline_kwargs: dict = field(default_factory=dict)

    @property
    def is_trainable(self) -> bool:
        return self.encoder is not None and self.head is not None

    @property
    def label(self) -> str:
        from .encoders import ENCODER_LABELS
        from .heads import HEAD_LABELS

        if self.is_trainable:
            enc = ENCODER_LABELS.get(self.encoder, self.encoder)
            hd = HEAD_LABELS.get(self.head, self.head)
            suffix = " [необучен]" if self.randomize_only else ""
            return f"{enc} + {hd}{suffix}"
        from .baselines import BASELINE_LABELS

        return BASELINE_LABELS.get(self.baseline, self.baseline or self.name)


# --------------------------------------------------------------------- core
CORE: tuple[Config, ...] = (
    # ---- протокольные контроли: валидация метрики, а не конкуренты ----
    Config(
        name="ctrl_random",
        group="control",
        baseline="random",
        rationale=(
            "Случайный score. Kim et al. (2021): при point-adjustment он выходит на уровень SOTA. "
            "Если здесь он попадает в верхнюю половину по основной метрике — метрика выбрана неверно "
            "и остальная таблица недействительна. Ожидается AP≈prevalence, padf≈0."
        ),
    ),
    Config(
        name="ctrl_untrained",
        group="control",
        encoder="gcn_gru",
        head="recon",
        randomize_only=True,
        rationale=(
            "Та же архитектура со случайными весами. У Kim et al. необученная модель оказалась "
            "сравнима с опубликованными методами даже без PA. Разность «обученная − эта» — "
            "измеренный вклад обучения."
        ),
    ),
    Config(
        name="base_pca",
        group="control",
        baseline="pca",
        rationale=(
            "Линейная нижняя граница. Sehili et al. (2023): PCA превосходит многие DL-подходы; "
            "Alves et al. (2026) подтвердили на SMD без point-adjustment. Если ST-GNN не "
            "отрывается от PCA значимо — это и есть результат."
        ),
    ),
    # ---- доменные и неглубокие бейзлайны ----
    Config(
        name="base_california",
        group="baseline",
        baseline="california",
        rationale=(
            "Канонический AID-алгоритм по пространственным разностям занятости. Обязателен для "
            "транспортного журнала: без него работа читается как ML-статья про трафик, "
            "а не как вклад в автоматическое обнаружение инцидентов."
        ),
    ),
    Config(
        name="base_snd",
        group="baseline",
        baseline="snd",
        rationale=(
            "Standard Normal Deviate — классический статистический контроль на детекторе. "
            "Вторая доменная точка отсчёта, не использующая пространственную структуру вообще."
        ),
    ),
    Config(
        name="base_iforest",
        group="baseline",
        baseline="iforest",
        rationale=(
            "Adaptive Isolation Forest даёт 71–100% detection rate на реальных городских событиях "
            "(J. Intelligent Transportation Systems, 2025). Включён как сильный неглубокий "
            "конкурент, а не для контраста."
        ),
    ),
    # ---- глубокий неграфовый: проверка вклада пространственного prior ----
    Config(
        name="deep_transformer_contrastive",
        group="non_graph",
        encoder="transformer",
        head="contrastive",
        rationale=(
            "TransDe-подобная связка: мультимасштабный контрастив поверх внимания, без графа. "
            "Лидирует на четырёх MTSAD-бенчмарках из пяти. Если обходит графовые модели — "
            "физический prior коридора не нужен."
        ),
    ),
    # ---- кандидаты ----
    Config(
        name="cand_gcngru_recon",
        group="candidate",
        encoder="gcn_gru",
        head="recon",
        rationale=(
            "РЕФЕРЕНС. Бенчмарк FT-AED (Coursey et al., 2024): unsupervised графовые "
            "автоэнкодеры — лучшее семейство на этих данных, а игнорирование пространственных "
            "связей ухудшает качество. Все остальные кандидаты сравниваются с этой точкой."
        ),
    ),
    Config(
        name="cand_gatlstm_recon",
        group="candidate",
        encoder="gat_lstm",
        head="recon",
        rationale=(
            "Обучаемое внимание вместо фиксированного графа при той же голове. В работе "
            "«Toward explainable automatic incident detection» (2025) GAT+LSTM — лучшая пара "
            "пространственного и временного модуля. Проверяет, размывает ли фиксированное "
            "сглаживание сам сигнал аномалии."
        ),
    ),
    Config(
        name="cand_gcngru_bigan",
        group="candidate",
        encoder="gcn_gru",
        head="bigan",
        rationale=(
            "Тот же энкодер, механизм score — критик. CCB-GraphGAN (Nouri et al., 2026) на FT-AED: "
            "−5 мин к задержке при FPR 1%. Одновременно проверяет открытый пробел: при D*=1/2 "
            "выход критика не несёт информации об аномалии (см. абляцию cycle vs critic)."
        ),
    ),
    Config(
        name="cand_hypergraph_flow",
        group="candidate",
        encoder="hypergraph",
        head="flow",
        rationale=(
            "DHMPN-подобная связка: направленный гиперграф + нормализующий поток. Явная плотность "
            "вместо эвристики, дешевле трансформеров, заявлено превосходство над SOTA-графами. "
            "Единственный кандидат с калиброванным вероятностным score — нужен для "
            "safety-валидатора в контуре устранения."
        ),
    ),
    Config(
        name="cand_gcngru_physics",
        group="candidate",
        encoder="gcn_gru",
        head="physics",
        rationale=(
            "Невязка закона сохранения LWR как score. НЕЗАНЯТАЯ НИША: physics-informed подход "
            "применён к обнаружению заторов и к генерации, но не как голова детектора. "
            "Единственный механизм, чей score не зависит от обученного генератора, "
            "то есть структурно невосприимчив к циркулярности оценки."
        ),
    ),
)

# ----------------------------------------------------------------- extended
_EXT_ENCODERS = ("gcn_gru", "gat_lstm", "hypergraph", "transformer", "gated_tcn")
_EXT_HEADS = ("recon", "flow", "bigan")

EXTENDED: tuple[Config, ...] = tuple(
    Config(
        name=f"ext_{e}_{h}",
        group="candidate" if e in {"gcn_gru", "gat_lstm", "hypergraph"} else "non_graph",
        encoder=e,
        head=h,
        rationale=(
            f"Клетка полного креста {e} × {h}. Нужна для разложения дисперсии: "
            "что объясняет больше разброса — ось энкодера или ось механизма score. "
            "Это и есть главный вопрос бенчмарка."
        ),
    )
    for e in _EXT_ENCODERS
    for h in _EXT_HEADS
)

GRIDS: dict[str, tuple[Config, ...]] = {
    "core": CORE,
    "extended": EXTENDED,
    "all": CORE + EXTENDED,
    "smoke": CORE[:4] + (CORE[7], CORE[10]),
}

GROUP_LABELS: dict[str, str] = {
    "control": "Протокольный контроль",
    "baseline": "Бейзлайн",
    "non_graph": "Без графа",
    "candidate": "Кандидат",
}


def get_grid(name: str) -> tuple[Config, ...]:
    if name not in GRIDS:
        raise ValueError(f"неизвестная сетка {name!r}; доступны: {sorted(GRIDS)}")
    return GRIDS[name]


#: Референсная конфигурация, с которой сравниваются все остальные.
REFERENCE = "cand_gcngru_recon"

#: ПРЕДЗАРЕГИСТРИРОВАННОЕ подмножество для теста Фридмана и Nemenyi: референс,
#: четыре остальных кандидата и единственный глубокий неграфовый метод (без
#: него R2 «нужен ли граф» не имел бы представителя в выводном тесте).
#:
#: Зачем. Критическая разница Nemenyi растёт как sqrt(k(k+1)/6N): при 12 методах
#: и 20 блоках CD = 3.73, при 6 методах — 1.69. Омнибусный тест по всей сетке
#: не обладает мощностью (на первом CV-прогоне CD превысила весь размах рангов).
#: Бейзлайны и протокольные контроли из сводной таблицы НЕ удаляются и из
#: прогона не исключаются: они остаются в ней описательно, а попарный bootstrap
#: против референса считается для всех конфигураций.
#:
#: Список зафиксирован до итогового прогона; менять его после просмотра чисел —
#: подгонка (CLAUDE.md, правило 11).
PRIMARY_COMPARISON: tuple[str, ...] = (
    "cand_gcngru_recon",
    "cand_gatlstm_recon",
    "cand_gcngru_bigan",
    "cand_hypergraph_flow",
    "cand_gcngru_physics",
    "deep_transformer_contrastive",
)
assert REFERENCE in PRIMARY_COMPARISON
assert set(PRIMARY_COMPARISON) <= {c.name for c in CORE}
