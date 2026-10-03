"""Визуализации отчёта и единая визуальная система."""
from .figures import (
    fig_critical_difference,
    fig_encoder_head_heatmap,
    fig_event_timeline,
    fig_metric_inflation,
    fig_operating_curves,
    fig_pareto,
    fig_pr_curves,
    fig_seed_variance,
    fig_spatial_prior_contribution,
)
from .theme import CATEGORICAL, CATEGORICAL_SAFE3, GROUP_STYLE, MARKERS, apply_theme, save

__all__ = [
    "apply_theme", "save", "CATEGORICAL", "CATEGORICAL_SAFE3", "GROUP_STYLE", "MARKERS",
    "fig_critical_difference", "fig_operating_curves", "fig_pr_curves",
    "fig_encoder_head_heatmap", "fig_metric_inflation", "fig_seed_variance",
    "fig_pareto", "fig_event_timeline", "fig_spatial_prior_contribution",
]
