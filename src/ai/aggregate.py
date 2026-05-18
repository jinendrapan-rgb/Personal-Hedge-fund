"""Combine the four analyzers into one ai_score and overlay it on the
factor composite.

Per the spec the overlay is OFF by default: the factor signal must be
judged on its own (Layer 3) before we A/B test whether AI helps. When
enabled, the AI contribution is clipped to ±2 standard deviations of the
cross-sectional composite so a single model call can't dominate the book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai.models import AnalysisReport


def combine_reports(reports: list[AnalysisReport]) -> float:
    """Mean of the analyzers' score adjustments, clipped to [-2, +2].

    Analyzers that produced no flags contribute 0 (already enforced in
    ``AnalysisReport.parse``), so silent analyzers don't dilute a strong
    signal toward zero unfairly — they're genuine abstentions.
    """
    if not reports:
        return 0.0
    vals = [r.score_adjustment for r in reports]
    return float(np.clip(np.mean(vals), -2.0, 2.0))


def apply_overlay(
    factor_scores: pd.DataFrame,
    ai_scores: dict[str, float],
    *,
    enabled: bool = False,
) -> pd.DataFrame:
    """Return a copy with an ``ai_score`` column and, if ``enabled``, a
    composite adjusted by the clipped AI overlay.

    ``factor_scores`` must have ``ticker`` and ``composite`` columns.
    """
    out = factor_scores.copy()
    out["ai_score"] = out["ticker"].map(ai_scores).fillna(0.0)
    if not enabled:
        return out  # overlay computed but NOT applied (spec default)

    sigma = out["composite"].std(ddof=0)
    cap = 2.0 * sigma if np.isfinite(sigma) and sigma > 0 else 0.0
    overlay = out["ai_score"].clip(-cap, cap) if cap > 0 else 0.0
    out["composite"] = out["composite"] + overlay
    return out.sort_values("composite", ascending=False).reset_index(drop=True)
