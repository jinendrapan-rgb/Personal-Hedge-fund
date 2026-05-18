"""Correlation / concentration monitor.

* any position pair with |corr| > 0.85  -> over-concentration flag
* hierarchical clusters (distance = 1-corr); any cluster holding > 20%
  of gross exposure -> concentration alert
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class ConcentrationReport:
    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    heavy_clusters: list[tuple[list[str], float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.high_corr_pairs and not self.heavy_clusters


def monitor(
    weights: pd.Series,
    returns_history: pd.DataFrame,
    *,
    pair_threshold: float = 0.85,
    cluster_gross_limit: float = 0.20,
    cluster_distance: float = 0.4,   # 1 - corr; ~corr>0.6 groups together
) -> ConcentrationReport:
    held = weights[weights != 0]
    cols = [t for t in held.index if t in returns_history.columns]
    rep = ConcentrationReport()
    if len(cols) < 2:
        return rep

    corr = returns_history[cols].dropna(how="any").corr()
    gross = held[cols].abs()
    gross_total = float(gross.sum()) or 1.0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            cij = float(corr.iloc[i, j])
            if abs(cij) > pair_threshold:
                rep.high_corr_pairs.append((cols[i], cols[j], round(cij, 3)))

    dist = np.array((1.0 - corr).clip(lower=0.0).values, dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=cluster_distance, criterion="distance")
    for lab in np.unique(labels):
        members = [cols[i] for i in range(len(cols)) if labels[i] == lab]
        if len(members) < 2:
            continue
        share = float(gross[members].sum()) / gross_total
        if share > cluster_gross_limit:
            rep.heavy_clusters.append((members, round(share, 3)))
    return rep
