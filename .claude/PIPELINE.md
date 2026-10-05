# Пайплайн агентов для прогона эксперимента

Четыре роли, одна точка входа, один принудительный гейт.

```
/run-experiment "<цель>"
        │
        ▼
experiment-orchestrator ─── уточняет цель, ведёт задачник, решает A/B
        ├──► experiment-runner   make test → прогон в фоне → лог, failures.csv
        ├──► results-analyst     R0 → мощность → санитарные → R1..R10   (read-only)
        └──► code-surgeon        только класс A, либо класс B с разрешением
                │
                ▼
        PreToolUse hook (.claude/hooks/guard_protocol.py)
        deny: reports/, data/, удаление контролей, force-push
        ask:  docs/DECISION_RULES.md, docs/PROTOCOL.md, metrics/, registry.py,
              data/, encoders/, heads/, итоговые конфиги, .claude/
```

## Почему именно так

- **Разделены роли «запустить» и «истолковать».** Runner без прав на правку
  не может «починить» прогон, подкрутив конфиг; analyst без прав на запись
  не может подогнать код под вывод.
- **Классификация A/B — в двух местах.** В промпте оркестратора (чтобы он
  спрашивал) и в hook (чтобы это нельзя было обойти, в том числе в auto-режиме).
  Промпт — договорённость, hook — механизм.
- **`permissionDecision: "ask"`** эскалирует вопрос пользователю даже когда
  сессия идёт без присмотра; `"deny"` возвращает агенту причину, и он видит,
  почему нельзя.
- **Аудит.** Каждое решение гейта пишется в `reports/agent_audit.jsonl`:
  потом видно, что менялось между двумя прогонами.

## Как запускать

```bash
cd F:\AITU\PhD\projects\stad-anomaly-detection-benchmark
claude
/run-experiment ft-aed-cv на cuda, закрыть R1 и R3
```

Варианты:

- `claude --agent experiment-orchestrator` — вся сессия идёт как оркестратор.
- `@"results-analyst (agent)" разбери reports/ft_aed_cv` — только анализ
  уже готового прогона, без перезапуска.
- `claude -p "/run-experiment smoke"` — headless, для cron/CI.

## Что настроить под себя

- `ASK_PATTERNS` / `DENY_PATTERNS` в `.claude/hooks/guard_protocol.py` — список
  протокольных файлов.
- `permissions.ask` в `.claude/settings.json` — тяжёлые цели make, которые не
  должны стартовать молча.
- `model` в frontmatter агентов: runner дешёвый (sonnet), analyst и surgeon — opus.
