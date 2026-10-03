.PHONY: help setup test smoke synthetic ft-aed ft-aed-extended figures clean lint

PY ?= python
PIP ?= pip

help:          ## показать доступные цели
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:         ## установить зависимости
	$(PIP) install -r requirements.txt

test:          ## прогнать тесты (метрики, модели, данные)
	$(PY) -m pytest tests -v

smoke:         ## быстрая проверка пайплайна на синтетике (минуты)
	$(PY) scripts/run_benchmark.py --config configs/smoke.yaml

synthetic:     ## полная сетка на синтетике: объяснение механизма
	$(PY) scripts/run_benchmark.py --config configs/synthetic_full.yaml

ft-aed:        ## ИТОГОВЫЙ прогон для диссертации: 12 конфигураций на FT-AED
	$(PY) scripts/run_benchmark.py --config configs/ft_aed_core.yaml

ft-aed-extended: ## полный крест энкодер × голова на FT-AED (дорого)
	$(PY) scripts/run_benchmark.py --config configs/ft_aed_extended.yaml

figures:       ## пересобрать фигуры и отчёт без обучения
	$(PY) scripts/make_figures.py --config configs/ft_aed_core.yaml

data:          ## скачать FT-AED и проверить схему
	$(PY) scripts/download_data.py --dataset ft-aed

lint:          ## проверить стиль (если установлен ruff)
	-ruff check src tests scripts

clean:         ## удалить кэши python
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
