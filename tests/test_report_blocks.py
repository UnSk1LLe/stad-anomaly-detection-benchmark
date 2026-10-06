"""Блокировки выводов в отчёте: R0 блокирует всё, R1 блокирует R2–R7, R3 — только на кресте.

Регрессия пилотных прогонов: R0 был провален, а разделы 3–5 RESULTS.md всё равно
печатали «главный результат» и «главный аргумент» (TZ_PROTOCOL_V2, §1.1).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stad.data import make_synthetic_corridor
from stad.experiment import ExperimentConfig
from stad.registry import CORE, EXTENDED, REFERENCE
from stad.report import (
    BLOCKED_R0,
    BLOCKED_R1,
    NEEDS_EXTENDED,
    decide,
    full_cross,
    rank_table,
    validate_protocol,
    write_results_md,
)
from stad.runner import run_grid
from stad.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {"prevalence": {"fold0": 0.3}, "n_seeds": 5, "alarm_budget_per_hour": 0.25}

# Уровни padf: кандидаты различимы (мощность есть), контроли у пола,
# референс заметно выше необученной версии.
LEVELS = {
    "ctrl_random": 0.05, "ctrl_untrained": 0.05, "base_pca": 0.10,
    "base_california": 0.15, "base_snd": 0.15, "base_iforest": 0.15,
    "deep_transformer_contrastive": 0.20, "cand_gcngru_physics": 0.25,
    REFERENCE: 0.30, "cand_gatlstm_recon": 0.35, "cand_gcngru_bigan": 0.40,
    "cand_hypergraph_flow": 0.45,
}


def _fake_runs(*, levels: dict[str, float] | None = None, extended: bool = False,
               n_folds: int = 4, n_seeds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Синтетический ``runs`` в формате раннера: ядро и, по желанию, клетки ``ext_*``."""
    levels = {**LEVELS, **(levels or {})}
    rng = np.random.default_rng(seed)
    configs = CORE + (EXTENDED if extended else ())
    rows = []
    for f in range(n_folds):
        for s in range(n_seeds):
            for i, c in enumerate(configs):
                level = levels.get(c.name, 0.2 + 0.01 * (i % 10))
                rows.append({
                    "config": c.name, "label": c.name, "group": c.group, "group_label": c.group,
                    "encoder": c.encoder, "head": c.head,
                    "uses_graph": None if c.encoder is None else c.encoder in {"gcn_gru", "gat_lstm",
                                                                                "hypergraph"},
                    "dataset": f"fold{f}", "seed": s, "block": f"fold{f}|s{s}",
                    "padf": level + rng.normal(0, 0.01),
                    "average_precision": 0.3, "prevalence": 0.3, "event_recall": 0.2,
                    "pa_f1_INVALID_for_ranking": 0.8, "budget_within_tolerance": True,
                    "alarm_budget_per_hour": 0.25, "alarms_per_hour": 0.25,
                    "ap_lift_over_random": 1.0, "fpr_observed": 0.01,
                    "inference_ms_per_window": 1.0, "median_delay_min": 0.0,
                    "n_params": 1000, "pa_inflation_ratio": 3.0,
                    "physics_correction_share": np.nan,
                })
    return pd.DataFrame(rows)


def _by_rule(rules: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["rule"].split(" ", 1)[0]: r for r in rules}


def _section(md: str, start: str, end: str) -> str:
    return md.split(start, 1)[1].split(end, 1)[0]


# ------------------------------------------------------------------- R0
def test_r0_failure_blocks_every_rule_and_report_sections(tmp_path):
    runs = _fake_runs(levels={"ctrl_random": 0.9})        # случайный контроль выше всех
    checks = validate_protocol(runs, prevalence=0.3)
    assert any(c.critical and not c.passed for c in checks)

    # checks не передаётся: decide обязан посчитать гейт сам
    rules = decide(runs)
    assert rules[0]["rule"].startswith("R0") and rules[0]["status"] == "failed"
    assert "СТОП" in rules[0]["action"]
    rest = rules[1:]
    assert {r["rule"].split()[0] for r in rest} >= {"R1", "R2", "R4", "R5", "R6"}
    for r in rest:
        assert r["action"] == BLOCKED_R0 and r["status"] == "blocked_r0"
        assert r["finding"].startswith("Справочно, без вывода: ")   # числа остаются

    md = write_results_md(runs, pd.DataFrame(), out_path=tmp_path / "RESULTS.md",
                          manifest=MANIFEST, figures=[]).read_text(encoding="utf-8")
    sec5 = _section(md, "## 5.", "## 6.")
    actions = [ln for ln in sec5.splitlines() if ln.startswith("**Что делать.**")]
    assert actions
    for ln in actions:
        assert ln == f"**Что делать.** {BLOCKED_R0}" or ln.startswith("**Что делать.** СТОП")
    assert "ГЛАВН" not in sec5 and "глав" not in sec5.lower()
    assert "`blocked_r0`" in sec5

    sec3 = _section(md, "## 3.", "## 4.")
    verdicts = [ln.rsplit("|", 2)[1].strip() for ln in sec3.splitlines()
                if ln.startswith("| ") and not ln.startswith("| Конфигурация")]
    assert verdicts and all(v == "не оценивается (R0)" for v in verdicts)
    assert "+0." in sec3 or "-0." in sec3                    # числовые колонки на месте


# ------------------------------------------------------------------- R1
def test_r1_not_significant_blocks_r2_to_r7():
    runs = _fake_runs()
    # PCA = референс по блокам ± 0.01 поочерёдно: средняя парная разность ровно ноль
    ref = runs.loc[runs["config"] == REFERENCE, "padf"].to_numpy()
    runs.loc[runs["config"] == "base_pca", "padf"] = ref + 0.01 * (-1.0) ** np.arange(len(ref))
    checks = validate_protocol(runs, prevalence=0.3)
    assert all(c.passed for c in checks if c.critical)

    rules = _by_rule(decide(runs, checks=checks))
    assert "отрыв НЕ значим" in rules["R1"]["finding"]
    assert rules["R1"]["status"] == "ok"
    assert rules["R1"]["action"].startswith("ГЛАВНЫЙ РЕЗУЛЬТАТ ДЛЯ ГЛАВЫ 6")   # исход самого R1
    blocked = [r for k, r in rules.items() if k in {"R2", "R3", "R4", "R5", "R6", "R7"}]
    assert len(blocked) >= 4
    for r in blocked:
        assert r["action"].startswith(BLOCKED_R1) and r["status"] == "blocked_r1"


def test_r1_not_evaluated_blocks_r2_to_r7():
    runs = _fake_runs()
    runs = runs[runs["config"] != "base_pca"]
    rules = _by_rule(decide(runs))
    assert "R1" not in rules
    for k in ("R2", "R4", "R5", "R6"):
        assert rules[k]["action"].startswith(BLOCKED_R1)
        assert "R1 в этом прогоне не оценён" in rules[k]["action"]


# ------------------------------------------------------------------- R3
def test_r3_requires_full_cross():
    runs = _fake_runs()                                       # R0 и R1 проходят
    rules = _by_rule(decide(runs))
    assert rules["R1"]["status"] == "ok" and "отрыв значим" in rules["R1"]["finding"]
    assert rules["R3"]["status"] == "needs_extended"
    assert rules["R3"]["action"].startswith(NEEDS_EXTENDED) and "`all`" in rules["R3"]["action"]
    assert "6 из 4×5 = 20" in rules["R3"]["finding"]
    assert full_cross(runs).empty
    assert rules["R2"]["status"] == "ok"                      # остальные правила не тронуты

    ext = _fake_runs(extended=True)
    cross = full_cross(ext)
    assert set(cross["config"]) == {c.name for c in EXTENDED}
    r3 = _by_rule(decide(ext))["R3"]
    assert r3["status"] == "ok" and "eta²" in r3["finding"] and "15 конфигураций" in r3["finding"]


def test_r3_gap_counts_extended_subset_and_single_level_axis():
    ext = _fake_runs(extended=True)
    r3 = _by_rule(decide(ext[ext["config"] != "ext_gated_tcn_bigan"]))["R3"]
    assert r3["status"] == "needs_extended"
    assert "14 из 5×3 = 15 пар" in r3["finding"] and "extended" in r3["finding"]

    runs = _fake_runs()
    keep = runs[runs["config"].isin([REFERENCE, "cand_gcngru_bigan", "cand_gcngru_physics"])]
    rest = runs[~runs["encoder"].notna() | (runs["group"] == "control")]
    r3 = _by_rule(decide(pd.concat([rest, keep], ignore_index=True)))["R3"]
    assert r3["status"] == "needs_extended"
    assert "один уровень" in r3["finding"] and "не является полным крестом" not in r3["finding"]


def test_full_cross_uses_only_complete_blocks():
    """Клетка креста, упавшая в одном блоке, не делает дизайн несбалансированным."""
    ext = _fake_runs(extended=True)
    drop = (ext["config"] == "ext_gated_tcn_bigan") & (ext["block"] == "fold0|s0")
    cross = full_cross(ext[~drop])
    assert set(cross["config"]) == {c.name for c in EXTENDED}
    assert "fold0|s0" not in set(cross["block"]) and cross["block"].nunique() == 19


def test_results_md_shows_r3_eta_when_cross_is_a_subset(tmp_path):
    """Раздел 4 по всем конфигурациям не должен расходиться с R3 молча."""
    ext = _fake_runs(extended=True)
    r3 = _by_rule(decide(ext))["R3"]
    md = write_results_md(ext, pd.DataFrame(), out_path=tmp_path / "RESULTS.md",
                          manifest=MANIFEST, figures=[]).read_text(encoding="utf-8")
    sec4 = _section(md, "## 4.", "## 5.")
    assert "описательная" in sec4 and "полном кресте из 15 конфигураций" in sec4
    enc = r3["finding"].split("eta² энкодера = ", 1)[1][:5]
    head = r3["finding"].split("eta² головы = ", 1)[1][:5]
    assert f"| энкодер | 5 | {enc} |" in sec4 and f"| механизм score (голова) | 3 | {head} |" in sec4


def test_full_cross_excludes_controls():
    """``ctrl_untrained`` (gcn_gru × recon) не достраивает крест."""
    runs = _fake_runs()
    keep = runs[runs["config"].isin(["cand_gcngru_bigan", "cand_gatlstm_recon", "ctrl_untrained"])]
    extra = keep[keep["config"] == "cand_gcngru_bigan"].assign(
        config="x_gat_bigan", encoder="gat_lstm", head="bigan")
    sub = pd.concat([keep, extra], ignore_index=True)
    assert full_cross(sub).empty
    with_ctrl_as_candidate = sub.assign(group="candidate")
    assert not full_cross(with_ctrl_as_candidate).empty


def test_results_md_notes_descriptive_eta_on_non_crossed_grid(tmp_path):
    md = write_results_md(_fake_runs(), pd.DataFrame(), out_path=tmp_path / "RESULTS.md",
                          manifest=MANIFEST, figures=[]).read_text(encoding="utf-8")
    sec4 = _section(md, "## 4.", "## 5.")
    assert "не является полным крестом" in sec4 and "описательная" in sec4


# ------------------------------------------------------------------- текст проверки
def test_padf_floor_detail_uses_run_budget():
    runs = _fake_runs()
    detail = {c.name: c.detail for c in validate_protocol(runs, prevalence=0.3)}[
        "Случайный score: padf у пола"]
    assert "0.25" in detail and "1 тревога/ч" not in detail

    mixed = pd.concat([runs, runs.assign(alarm_budget_per_hour=1.0)], ignore_index=True)
    detail = {c.name: c.detail for c in validate_protocol(mixed, prevalence=0.3)}[
        "Случайный score: padf у пола"]
    assert "0.25, 1" in detail

    detail = {c.name: c.detail for c in validate_protocol(runs, 0.3, alarm_budget_per_hour=0.5)}[
        "Случайный score: padf у пола"]
    assert "0.5" in detail


# ------------------------------------------------------------------- время
def test_timing_star_for_partially_timed_configs(tmp_path):
    """Старый runs.csv (без timing_source): у возобновлённых клеток время NaN."""
    runs = _fake_runs()
    ref = runs["config"] == REFERENCE
    runs.loc[ref & (runs["seed"] < 3), "inference_ms_per_window"] = np.nan      # 8 из 20
    runs.loc[runs["config"] == "cand_gcngru_bigan", "inference_ms_per_window"] = np.nan
    assert "timing_source" not in runs.columns

    table = rank_table(runs).set_index("config")
    assert table.loc[REFERENCE, "n_timed"] == 8 and table.loc[REFERENCE, "n_runs"] == 20
    assert table.loc["cand_gcngru_bigan", "n_timed"] == 0
    assert table.loc["base_pca", "n_timed"] == 20

    md = write_results_md(runs, pd.DataFrame(), out_path=tmp_path / "RESULTS.md",
                          manifest=MANIFEST, figures=[]).read_text(encoding="utf-8")
    row = {ln.split("|")[1].strip(): ln.rstrip(" |").rsplit("|", 1)[1].strip()
           for ln in _section(md, "## 2.", "## 3.").splitlines()
           if ln.startswith("| ") and not ln.startswith("| Конфигурация")}
    assert row[REFERENCE] == "1.00*"
    assert row["cand_gcngru_bigan"] == "—"
    assert row["base_pca"] == "1.00"
    assert "среднее по k из n клеток" in md and f"{REFERENCE} — 8 из 20" in md
    assert "cand_gcngru_bigan — 0 из" not in md
    assert "нет ни в одной клетке: cand_gcngru_bigan" in md


# ------------------------------------------------------------------- пересборка
def _load_script():
    spec = importlib.util.spec_from_file_location("rebuild_report", ROOT / "scripts" / "rebuild_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rebuild_report_from_run_directory(tmp_path):
    data = make_synthetic_corridor(stations=5, lanes=3, days=3, step_min=1.0,
                                   n_events=8, window=8, seed=2)
    names = {"ctrl_random", "base_pca", "base_snd", REFERENCE}
    configs = tuple(c for c in CORE if c.name in names)
    run_dir = tmp_path / "run"
    run_grid(configs, {"synthetic": data}, seeds=(0, 1), param_budget=20_000, out_dir=run_dir,
             train_cfg=TrainConfig(epochs=1, batch_size=64, device="cpu"),
             save_checkpoints=False, verbose=False)
    cfg = ExperimentConfig(name="t", grid="smoke", seeds=(0, 1), param_budget=20_000,
                           out_dir=str(run_dir))
    (run_dir / "experiment_config.json").write_text(
        json.dumps({**cfg.__dict__, "seeds": list(cfg.seeds)}, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    sha = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["environment"]["git_sha"]

    script = _load_script()
    out = tmp_path / "rebuilt"
    assert script.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert md.splitlines()[2].startswith(f"Пересобрано из `{run_dir.as_posix()}` (прогон коммита `{sha}`)")
    assert "ТЗ v2, задача 1.3" in md and "Сгенерировано автоматически" not in md
    figs = {p.name.split("_")[0] for p in (out / "figures").glob("*.png")}
    assert {"fig01", "fig02"} <= figs and not figs & {"fig03", "fig08"}
    assert any((out / "tables").glob("*.csv"))
    assert not (run_dir / "RESULTS.md").exists() and not (run_dir / "figures").exists()

    assert script.main(["--run-dir", str(run_dir), "--out", str(run_dir)]) == 2
    assert script._shown(ROOT / "reports" / "ft_aed_cv") == "reports/ft_aed_cv"   # без локального пути
