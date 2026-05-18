"""Parse SEC ``companyfacts`` XBRL into a tidy frame and enforce the
point-in-time contract.

The non-negotiable rule for this whole system: a fundamental must be
retrievable *as it was known on a given date*, never as later restated.
Every XBRL fact carries the ``filed`` date of the filing that reported it,
so ``as_of`` filters ``filed <= as_of_date`` and then takes the most
recently-filed value for the most recent reporting period known by then.

Concretely, for a metric first reported X then restated to Y:

* ``as_of`` *before* the restatement's filing date  -> X  (original)
* ``as_of`` *after*  the restatement's filing date  -> Y  (restated)
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

_TIDY_COLS = [
    "taxonomy", "concept", "unit", "val",
    "start", "end", "fy", "fp", "form", "filed", "accn", "frame",
]

# Common concepts the factor engine will need, with fallback tag names.
STANDARD_CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "net_income": ["NetIncomeLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "dei:EntityCommonStockSharesOutstanding"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
}


def flatten(facts: dict[str, Any]) -> pd.DataFrame:
    """Flatten a ``companyfacts`` JSON document into one tidy DataFrame.

    Empty (correctly-typed) frame if there are no facts.
    """
    rows: list[dict[str, Any]] = []
    for taxonomy, concepts in (facts.get("facts") or {}).items():
        for concept, body in concepts.items():
            for unit, observations in (body.get("units") or {}).items():
                for obs in observations:
                    rows.append(
                        {
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "unit": unit,
                            "val": obs.get("val"),
                            "start": obs.get("start"),
                            "end": obs.get("end"),
                            "fy": obs.get("fy"),
                            "fp": obs.get("fp"),
                            "form": obs.get("form"),
                            "filed": obs.get("filed"),
                            "accn": obs.get("accn"),
                            "frame": obs.get("frame"),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=_TIDY_COLS)
    df = pd.DataFrame(rows, columns=_TIDY_COLS)
    df["filed"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"])
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    return df


def _as_ts(d: str | dt.date | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _annual_subset(grp: pd.DataFrame) -> pd.DataFrame:
    """Rows representing annual figures for one concept.

    Instant concepts (balance sheet: no ``start``) have no duration, so
    every observation is a point-in-time snapshot — keep them all. Flow
    concepts (income/cash-flow: have ``start``) repeat as 3-month *and*
    12-month figures with overlapping ``end`` dates. Period *duration* is
    the only reliable annual discriminator: ``fp`` is frequently "FY" even
    on the 3-month Q4 value pulled from an 8-K, which is exactly how a
    stale quarter sneaks past and gets chosen as "latest filed". So filter
    flows to a ~365-day duration first; only fall back to ``fp``/all if a
    concept genuinely has no annual observation.
    """
    instant = grp[grp["start"].isna()]
    flow = grp[grp["start"].notna()]
    if flow.empty:
        return instant
    days = (flow["end"] - flow["start"]).dt.days
    annual_flow = flow[(days >= 300) & (days <= 400)]
    if annual_flow.empty:
        annual_flow = flow[flow["fp"] == "FY"]
    if annual_flow.empty:
        annual_flow = flow  # last resort: don't drop the concept entirely
    return pd.concat([instant, annual_flow])


def as_of(
    tidy: pd.DataFrame,
    as_of_date: str | dt.date | pd.Timestamp,
    concept: str | None = None,
    period: str = "annual",
) -> pd.DataFrame:
    """Resolve each concept to its point-in-time value as of ``as_of_date``.

    For every ``(taxonomy, concept, unit)``:

    1. keep only facts with ``filed <= as_of_date`` (nothing from the future);
    2. if ``period == "annual"``, restrict flow concepts to annual figures
       so periods stay consistent across concepts (see ``_annual_subset``);
    3. restrict to the most recent reporting period (max ``end``) known by then;
    4. within that period, take the most recently-filed value (this is what
       makes a restatement resolve correctly on either side of its filing).

    Returns one row per concept with the resolved value.
    """
    if tidy.empty:
        return tidy.copy()
    asof = _as_ts(as_of_date)
    df = tidy[tidy["filed"] <= asof]
    if concept is not None:
        df = df[df["concept"] == concept]
    if df.empty:
        return df.copy()

    resolved: list[pd.DataFrame] = []
    for _, grp in df.groupby(["taxonomy", "concept", "unit"], sort=False):
        g = _annual_subset(grp) if period == "annual" else grp
        if g.empty:
            continue
        latest_period = g["end"].max()
        period_rows = g[g["end"] == latest_period]
        # Most recently filed value for that period == latest knowledge.
        best = period_rows.sort_values("filed").iloc[[-1]]
        resolved.append(best)
    if not resolved:
        return df.iloc[0:0].copy()
    return pd.concat(resolved, ignore_index=True).sort_values("concept")


def fundamentals_as_of(
    tidy: pd.DataFrame,
    as_of_date: str | dt.date | pd.Timestamp,
) -> dict[str, float | None]:
    """Convenience: standard concept set resolved to a flat dict.

    Uses ``STANDARD_CONCEPTS`` with fallbacks; missing concepts -> ``None``.
    """
    snap = as_of(tidy, as_of_date)
    if snap.empty:
        return {name: None for name in STANDARD_CONCEPTS}
    row_by_concept = {
        r["concept"]: r for _, r in snap.iterrows() if pd.notna(r["val"])
    }
    out: dict[str, float | None] = {}
    for name, candidates in STANDARD_CONCEPTS.items():
        # A company may carry an old tag (e.g. us-gaap:Revenues) frozen at
        # an old fiscal year alongside the current one (Revenue606). Pick
        # the candidate whose data is the most recent fiscal year; break
        # ties by candidate order (declared preference).
        best_val: float | None = None
        best_end = None
        for pref, tag in enumerate(candidates):
            tag = tag.split(":")[-1]
            r = row_by_concept.get(tag)
            if r is None:
                continue
            key = (r["end"], -pref)
            if best_end is None or key > best_end:
                best_end = key
                best_val = float(r["val"])
        out[name] = best_val
    return out
