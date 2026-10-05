"""PreToolUse-гейт пайплайна эксперимента.

Решает за Claude Code, можно ли выполнить вызов инструмента:

  deny  — действие делает результаты недействительными (ручная правка reports/,
          удаление протокольных контролей, force-push, rm -rf);
  ask   — существенное изменение протокола/плана/архитектуры: спрашивать
          пользователя, даже если сессия идёт в auto-режиме;
  (0)   — молчание: обычный порядок разрешений.

Все решения дописываются в reports/agent_audit.jsonl, чтобы потом было видно,
кто и что менял между прогонами.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

# --- файлы, менять которые = менять смысл эксперимента (класс B) -------------
ASK_PATTERNS = [
    r"^docs/decision_rules\.md$",
    r"^docs/protocol\.md$",
    r"^claude\.md$",
    r"^src/stad/metrics/",
    r"^src/stad/registry\.py$",
    r"^src/stad/report/decide",
    r"^src/stad/data/",            # разбиение, фолды, evaluation_view
    r"^src/stad/encoders/",        # новая/изменённая архитектура
    r"^src/stad/heads/",
    r"^configs/ft_aed_cv\.yaml$",
    r"^configs/ft_aed_extended\.yaml$",
    r"^configs/ft_aed_core\.yaml$",
    r"^configs/data/",
    r"^\.claude/",                 # сам пайплайн агентов
]

# --- файлы, которые агент не правит руками вообще ----------------------------
DENY_PATTERNS = [
    (r"^reports/", "Результаты получаются прогоном, а не правкой файла. "
                   "Перезапусти прогон (make ...) вместо ручного изменения reports/."),
    (r"^data/",    "Слой данных не правится вручную: воспроизводимость прогона "
                   "ломается молча. Используй scripts/download_data.py."),
]

# --- протокольные контроли: их удаление запрещено правилом 4 из CLAUDE.md ----
CONTROL_TOKENS = ("ctrl_random", "ctrl_untrained", "base_pca")

# --- команды, которые гейт не пропускает ------------------------------------
BASH_DENY = [
    (r"\bgit\s+push\b.*(--force|-f)\b", "force-push в репозиторий эксперимента запрещён."),
    (r"\brm\s+-rf\b", "рекурсивное удаление запрещено: артефакты прогонов невосстановимы."),
    (r"\bgit\s+checkout\s+--\s+", "сброс изменений целиком скрывает, что было сделано; "
                                  "откатывай точечно и объясняй."),
]
BASH_ASK = [
    (r"\bgit\s+push\b", "push в удалённый репозиторий"),
    (r"\bgit\s+reset\b", "сброс истории"),
]


def rel(path: str, project: str) -> str:
    p = (path or "").replace("\\", "/")
    pr = (project or "").replace("\\", "/").rstrip("/")
    if pr and p.lower().startswith(pr.lower() + "/"):
        p = p[len(pr) + 1:]
    return p.lstrip("./").lower()


def decide(event: dict) -> tuple[str | None, str]:
    tool = event.get("tool_name", "")
    ti = event.get("tool_input", {}) or {}
    project = event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", "")

    if tool == "Bash":
        cmd = ti.get("command", "") or ""
        for pat, why in BASH_DENY:
            if re.search(pat, cmd):
                return "deny", why
        for pat, why in BASH_ASK:
            if re.search(pat, cmd):
                return "ask", f"Подтверди: {why}."
        return None, ""

    if tool not in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        return None, ""

    path = rel(ti.get("file_path") or ti.get("notebook_path") or "", project)
    payload = " ".join(
        str(ti.get(k, "")) for k in ("old_string", "new_string", "content", "new_source")
    )

    for pat, why in DENY_PATTERNS:
        if re.search(pat, path):
            return "deny", why

    # удаление контроля из конфигурации или реестра
    old = str(ti.get("old_string", ""))
    new = str(ti.get("new_string", ""))
    if old and any(t in old and t not in new for t in CONTROL_TOKENS):
        return "deny", ("Правило 4 (CLAUDE.md): протокольные контроли "
                        "ctrl_random / ctrl_untrained / base_pca не удаляются. "
                        "Если контроль поднялся в таблице — дефект в метрике, чини метрику.")

    if "pa_f1_INVALID_for_ranking" in payload and "rank" in payload.lower():
        return "ask", ("Похоже на использование point-adjustment F1 для ранжирования "
                       "(правило 1 CLAUDE.md). Подтверди, что это только фигура 05.")

    for pat in ASK_PATTERNS:
        if re.search(pat, path):
            return "ask", (f"`{path}` входит в протокольный контур эксперимента "
                           "(протокол / метрика / состав сетки / данные / архитектура). "
                           "Это существенное изменение плана — нужно твоё решение.")
    return None, ""


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except Exception:
        return 0

    verdict, reason = decide(event)

    try:
        project = event.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", ".")
        log = os.path.join(project, "reports", "agent_audit.jsonl")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                "session": event.get("session_id"),
                "tool": event.get("tool_name"),
                "target": (event.get("tool_input") or {}).get("file_path")
                          or (event.get("tool_input") or {}).get("command"),
                "decision": verdict or "pass",
                "reason": reason,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass

    if verdict:
        json.dump({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": verdict,
            "permissionDecisionReason": reason,
        }}, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
