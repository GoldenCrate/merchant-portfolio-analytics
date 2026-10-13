"""
analytics.py
============
Pure pandas / numpy analytics for a Visa Merchant Business Development portfolio.

NO Streamlit (or any UI) imports live here. Everything is a plain function of
DataFrames / numbers so it can be unit-tested in isolation and reused from a
notebook, a batch job, or the Streamlit app.

The analytical vocabulary mirrors how a Merchant BD analyst actually works:

  * Portfolio coverage - how much volume sits on Visa today, and where the
    whitespace (un-penetrated volume) is.
  * Account prioritisation - a transparent score that pushes HIGH-volume /
    LOW-penetration accounts with open opportunities and stale engagement to
    the top of the call list.
  * Pipeline - probability-weighted value of open opportunities by stage and by
    opportunity type.
  * Coverage gaps - accounts that have gone quiet (stale outreach) or whose key
    contact has lapsed and needs replacing.

Conventions
-----------
``merchants`` is one row per merchant with columns produced by
``data_gen.generate_data()``:
    merchant_id, merchant_name, industry, region, annual_payment_volume,
    visa_penetration, relationship_tier, last_outreach_date, outreach_count_ytd,
    contact_count, key_contact_active, engagement_score, archetype,
    debit_routing, co_brand, value_added_services, acceptance_solutions

``opportunities`` is one row per pipeline opportunity with columns:
    opportunity_id, merchant_id, type, stage, est_annual_value, probability,
    next_step, created_date
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OPPORTUNITY_FLAGS = [
    "debit_routing",
    "co_brand",
    "value_added_services",
    "acceptance_solutions",
]

OPEN_STAGES = ["Identified", "Qualified", "Developing", "Proposed"]
STAGE_ORDER = ["Identified", "Qualified", "Developing", "Proposed", "Won", "Lost"]


# ---------------------------------------------------------------------------
# Primitive metric formulas (scalar- and vector-friendly; hand-computable)
# ---------------------------------------------------------------------------

def coverage_rate(volume, visa_penetration) -> float:
    """Business question: what share of portfolio payment volume is already on Visa?

    coverage = sum(volume * penetration) / sum(volume)  (volume-weighted).
    Returns 0.0 when there is no volume. Accepts scalars or array-likes.
    """
    vol = np.asarray(volume, dtype=float)
    pen = np.asarray(visa_penetration, dtype=float)
    total = vol.sum()
    if total <= 0:
        return 0.0
    return float((vol * pen).sum() / total)


def whitespace_volume(volume, visa_penetration):
    """Business question: how much volume is NOT yet on Visa (the growth whitespace)?

    whitespace = volume * (1 - penetration), element-wise. Returns a numpy array
    (or scalar) of un-penetrated dollars.
    """
    vol = np.asarray(volume, dtype=float)
    pen = np.asarray(visa_penetration, dtype=float)
    return vol * (1.0 - pen)


def weighted_pipeline_value(est_annual_value, probability) -> float:
    """Business question: what is the probability-weighted value of a set of opportunities?

    weighted = sum(est_annual_value * probability). Accepts scalars or arrays.
    """
    val = np.asarray(est_annual_value, dtype=float)
    prob = np.asarray(probability, dtype=float)
    return float((val * prob).sum())


def _minmax(s: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]. A constant series maps to all-zeros (no signal)."""
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

def portfolio_summary(merchants: pd.DataFrame, opportunities: pd.DataFrame | None = None) -> dict:
    """Business question: what are the headline numbers for the whole book?

    Returns a dict:
      n_merchants            - number of merchant relationships.
      total_volume           - total annual payment volume ($).
      avg_visa_penetration   - simple mean of per-merchant Visa penetration.
      coverage_rate          - volume-weighted share of volume already on Visa.
      whitespace_volume      - total un-penetrated volume ($).
      n_open_opportunities   - count of opportunities in an open stage.
      weighted_pipeline      - probability-weighted value of open opportunities.
      by_industry            - DataFrame: merchants / volume / penetration per industry.
    """
    if merchants.empty:
        return {
            "n_merchants": 0,
            "total_volume": 0.0,
            "avg_visa_penetration": 0.0,
            "coverage_rate": 0.0,
            "whitespace_volume": 0.0,
            "n_open_opportunities": 0,
            "weighted_pipeline": 0.0,
            "by_industry": pd.DataFrame(
                columns=["industry", "merchants", "volume", "avg_penetration", "whitespace"]
            ),
        }

    total_volume = float(merchants["annual_payment_volume"].sum())
    ws = float(whitespace_volume(
        merchants["annual_payment_volume"], merchants["visa_penetration"]
    ).sum())

    by_industry = (
        merchants.assign(
            _ws=whitespace_volume(
                merchants["annual_payment_volume"], merchants["visa_penetration"]
            )
        )
        .groupby("industry", observed=True)
        .agg(
            merchants=("merchant_id", "nunique"),
            volume=("annual_payment_volume", "sum"),
            avg_penetration=("visa_penetration", "mean"),
            whitespace=("_ws", "sum"),
        )
        .reset_index()
        .sort_values("volume", ascending=False)
        .reset_index(drop=True)
    )

    n_open = 0
    weighted = 0.0
    if opportunities is not None and not opportunities.empty:
        open_opps = opportunities[opportunities["stage"].isin(OPEN_STAGES)]
        n_open = int(len(open_opps))
        weighted = weighted_pipeline_value(
            open_opps["est_annual_value"], open_opps["probability"]
        )

    return {
        "n_merchants": int(merchants["merchant_id"].nunique()),
        "total_volume": total_volume,
        "avg_visa_penetration": float(merchants["visa_penetration"].mean()),
        "coverage_rate": coverage_rate(
            merchants["annual_payment_volume"], merchants["visa_penetration"]
        ),
        "whitespace_volume": ws,
        "n_open_opportunities": n_open,
        "weighted_pipeline": weighted,
        "by_industry": by_industry,
    }


# ---------------------------------------------------------------------------
# Account prioritisation - the key deliverable
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "volume": 0.35,         # bigger books of volume matter more
    "whitespace": 0.30,     # LOW penetration = more room to grow Visa share
    "opportunity": 0.20,    # open eligibility flags = concrete things to sell
    "engagement_gap": 0.15,  # stale / disengaged accounts are being neglected
}


def prioritization_score(
    merchants: pd.DataFrame,
    weights: dict | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Business question: which accounts should the analyst work first?

    Builds a transparent 0-100 priority score. An account scores HIGH when it has
    high payment volume, LOW Visa penetration (commercial whitespace), several
    open opportunity flags, and stale / weak engagement. Each component is
    min-max scaled across the portfolio and combined with ``weights``.

    Components (each in [0, 1]):
      * volume        - min-max of annual_payment_volume.
      * whitespace    - min-max of volume * (1 - penetration), the un-penetrated $.
      * opportunity   - share of the four eligibility flags that are open (0..1).
      * engagement_gap- average of (min-max days since outreach) and
                        (1 - engagement_score/100); higher = more neglected.

    Returns every merchant with the component columns, ``priority_score``
    (0-100), and ``priority_rank`` (1 = highest priority), sorted by score desc.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    if merchants.empty:
        cols = list(merchants.columns) + [
            "whitespace_dollars", "open_opportunities", "days_since_outreach",
            "c_volume", "c_whitespace", "c_opportunity", "c_engagement_gap",
            "priority_score", "priority_rank",
        ]
        return pd.DataFrame(columns=cols)

    if as_of is None:
        as_of = pd.to_datetime(merchants["last_outreach_date"]).max()

    df = merchants.copy()

    # Raw signals.
    df["whitespace_dollars"] = whitespace_volume(
        df["annual_payment_volume"], df["visa_penetration"]
    )
    df["open_opportunities"] = df[OPPORTUNITY_FLAGS].sum(axis=1).astype(int)
    df["days_since_outreach"] = (
        (pd.to_datetime(as_of) - pd.to_datetime(df["last_outreach_date"])).dt.days
    ).clip(lower=0)

    # Scaled components.
    df["c_volume"] = _minmax(df["annual_payment_volume"])
    df["c_whitespace"] = _minmax(df["whitespace_dollars"])
    df["c_opportunity"] = df["open_opportunities"] / float(len(OPPORTUNITY_FLAGS))
    recency = _minmax(df["days_since_outreach"])
    engagement = 1.0 - (df["engagement_score"].astype(float) / 100.0).clip(0, 1)
    df["c_engagement_gap"] = (0.5 * recency + 0.5 * engagement)

    score = (
        w["volume"] * df["c_volume"]
        + w["whitespace"] * df["c_whitespace"]
        + w["opportunity"] * df["c_opportunity"]
        + w["engagement_gap"] * df["c_engagement_gap"]
    )
    df["priority_score"] = (100.0 * score).round(1)

    df = df.sort_values("priority_score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["priority_rank"] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Opportunity pipeline
# ---------------------------------------------------------------------------

def pipeline_summary(opportunities: pd.DataFrame) -> dict:
    """Business question: how healthy is the opportunity pipeline, by stage and type?

    Computes probability-weighted value (est_annual_value * probability) and rolls
    it up two ways. "Open" = stages Identified / Qualified / Developing / Proposed
    (Won and Lost are terminal and excluded from open pipeline totals).

    Returns a dict:
      by_stage     - DataFrame per stage: count, est_value, weighted_value
                     (ordered Identified -> Lost).
      by_type      - DataFrame per opportunity type: count, est_value,
                     weighted_value (open opportunities only), sorted by weighted.
      total_weighted_open - weighted value of all OPEN opportunities.
      total_est_open      - raw est value of all OPEN opportunities.
      n_open              - count of open opportunities.
      win_rate            - Won / (Won + Lost) by count (0.0 if none closed).
    """
    empty_stage = pd.DataFrame(columns=["stage", "count", "est_value", "weighted_value"])
    empty_type = pd.DataFrame(columns=["type", "count", "est_value", "weighted_value"])
    if opportunities is None or opportunities.empty:
        return {
            "by_stage": empty_stage,
            "by_type": empty_type,
            "total_weighted_open": 0.0,
            "total_est_open": 0.0,
            "n_open": 0,
            "win_rate": 0.0,
        }

    o = opportunities.copy()
    o["weighted_value"] = o["est_annual_value"].astype(float) * o["probability"].astype(float)

    by_stage = (
        o.groupby("stage", observed=True)
        .agg(count=("opportunity_id", "count"),
             est_value=("est_annual_value", "sum"),
             weighted_value=("weighted_value", "sum"))
        .reset_index()
    )
    # Order stages along the funnel.
    by_stage["stage"] = pd.Categorical(by_stage["stage"], categories=STAGE_ORDER, ordered=True)
    by_stage = by_stage.sort_values("stage").reset_index(drop=True)
    by_stage["stage"] = by_stage["stage"].astype(str)

    open_o = o[o["stage"].isin(OPEN_STAGES)]
    by_type = (
        open_o.groupby("type", observed=True)
        .agg(count=("opportunity_id", "count"),
             est_value=("est_annual_value", "sum"),
             weighted_value=("weighted_value", "sum"))
        .reset_index()
        .sort_values("weighted_value", ascending=False)
        .reset_index(drop=True)
    )

    n_won = int((o["stage"] == "Won").sum())
    n_lost = int((o["stage"] == "Lost").sum())
    win_rate = (n_won / (n_won + n_lost)) if (n_won + n_lost) > 0 else 0.0

    return {
        "by_stage": by_stage,
        "by_type": by_type,
        "total_weighted_open": float(open_o["weighted_value"].sum()),
        "total_est_open": float(open_o["est_annual_value"].sum()),
        "n_open": int(len(open_o)),
        "win_rate": float(win_rate),
    }


def opportunity_breakdown(opportunities: pd.DataFrame, open_only: bool = True) -> pd.DataFrame:
    """Business question: where does opportunity value concentrate by solution type?

    Returns one row per opportunity ``type`` with count, total est_annual_value,
    and weighted_value, sorted by weighted_value descending. By default counts
    only OPEN opportunities; set ``open_only=False`` to include Won/Lost.
    """
    if opportunities is None or opportunities.empty:
        return pd.DataFrame(columns=["type", "count", "est_value", "weighted_value"])

    o = opportunities.copy()
    if open_only:
        o = o[o["stage"].isin(OPEN_STAGES)]
    o = o.assign(
        weighted_value=o["est_annual_value"].astype(float) * o["probability"].astype(float)
    )
    out = (
        o.groupby("type", observed=True)
        .agg(count=("opportunity_id", "count"),
             est_value=("est_annual_value", "sum"),
             weighted_value=("weighted_value", "sum"))
        .reset_index()
        .sort_values("weighted_value", ascending=False)
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Engagement & coverage gaps
# ---------------------------------------------------------------------------

def coverage_gaps(
    merchants: pd.DataFrame,
    stale_days: int = 90,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Business question: which relationships have gone quiet or lost their key contact?

    Flags a merchant when EITHER its last outreach is older than ``stale_days``
    OR its key contact is inactive (``key_contact_active == False``). Adds:
      * days_since_outreach
      * stale_outreach       (bool)
      * needs_contact         (bool - key contact inactive, replacement needed)
      * gap_reason            (human-readable: which condition(s) tripped)
    Returns only flagged merchants, sorted by annual_payment_volume descending
    (biggest relationships at risk first).
    """
    if merchants.empty:
        cols = list(merchants.columns) + [
            "days_since_outreach", "stale_outreach", "needs_contact", "gap_reason"
        ]
        return pd.DataFrame(columns=cols)

    if as_of is None:
        as_of = pd.to_datetime(merchants["last_outreach_date"]).max()

    df = merchants.copy()
    df["days_since_outreach"] = (
        (pd.to_datetime(as_of) - pd.to_datetime(df["last_outreach_date"])).dt.days
    ).clip(lower=0)
    df["stale_outreach"] = df["days_since_outreach"] > stale_days
    df["needs_contact"] = ~df["key_contact_active"].astype(bool)

    flagged = df[df["stale_outreach"] | df["needs_contact"]].copy()

    def _reason(row) -> str:
        parts = []
        if row["needs_contact"]:
            parts.append("key contact inactive - replace")
        if row["stale_outreach"]:
            parts.append(f"no outreach in {int(row['days_since_outreach'])}d")
        return "; ".join(parts)

    if not flagged.empty:
        flagged["gap_reason"] = flagged.apply(_reason, axis=1)
    else:
        flagged["gap_reason"] = pd.Series(dtype="object")

    flagged = flagged.sort_values("annual_payment_volume", ascending=False).reset_index(drop=True)
    return flagged


if __name__ == "__main__":
    from data_gen import generate_data

    data = generate_data()
    merchants, opps = data["merchants"], data["opportunities"]

    summ = portfolio_summary(merchants, opps)
    print("Portfolio summary:")
    for k, v in summ.items():
        if isinstance(v, pd.DataFrame):
            continue
        print(f"  {k}: {v:,.3f}" if isinstance(v, float) else f"  {k}: {v}")

    pri = prioritization_score(merchants)
    print("\nTop 10 priority accounts (archetype should be whitespace-heavy):")
    print(pri.head(10)[["merchant_id", "merchant_name", "archetype",
                        "annual_payment_volume", "visa_penetration",
                        "open_opportunities", "priority_score"]].to_string(index=False))

    pipe = pipeline_summary(opps)
    print("\nPipeline by stage:")
    print(pipe["by_stage"].to_string(index=False))
    print(f"\nWeighted open pipeline: {pipe['total_weighted_open']:,.0f}  win rate: {pipe['win_rate']:.0%}")

    gaps = coverage_gaps(merchants)
    print(f"\nCoverage gaps flagged: {len(gaps)} merchants")
