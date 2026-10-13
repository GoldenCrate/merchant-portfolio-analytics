"""
test_analytics.py
=================
Unit tests for the merchant-portfolio analytics in ``analytics.py``.

Two layers of testing:

1. Hand-computable fixtures for the primitive formulas (coverage rate,
   whitespace volume, weighted pipeline) and for ``prioritization_score`` /
   ``pipeline_summary`` / ``coverage_gaps`` on tiny tables whose every number is
   worked out by hand in the test.

2. Ground-truth validation against the full synthetic dataset: the generator
   injects known archetypes (whitespace / well_covered / lapsed_contact /
   steady), and we assert that prioritisation surfaces the whitespace accounts,
   that every lapsed-contact account is flagged for a contact replacement, and
   that the pipeline weighting math is internally consistent.

Run:  python -m pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analytics as A
from analytics import OPEN_STAGES
from data_gen import generate_data, STAGE_PROBABILITY


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """The full synthetic dataset, generated once per test module (seeded)."""
    return generate_data(seed=42)


@pytest.fixture(scope="module")
def merchants(dataset):
    return dataset["merchants"]


@pytest.fixture(scope="module")
def opportunities(dataset):
    return dataset["opportunities"]


@pytest.fixture()
def tiny_merchants():
    """Three hand-built merchants with a clear priority ordering.

    A: huge volume, low penetration, all 4 flags open, very stale  -> top.
    B: medium volume, mid penetration, 1 flag, recent              -> middle.
    C: small volume, high penetration, 0 flags, recent, active     -> bottom.
    """
    as_of = pd.Timestamp("2026-10-13")
    rows = [
        {
            "merchant_id": "A", "merchant_name": "Alpha", "industry": "Retail",
            "region": "West", "annual_payment_volume": 100_000_000.0,
            "visa_penetration": 0.30, "relationship_tier": "Strategic",
            "last_outreach_date": as_of - pd.Timedelta(days=200),
            "outreach_count_ytd": 0, "contact_count": 2, "key_contact_active": True,
            "engagement_score": 20.0, "archetype": "whitespace",
            "debit_routing": True, "co_brand": True,
            "value_added_services": True, "acceptance_solutions": True,
        },
        {
            "merchant_id": "B", "merchant_name": "Beta", "industry": "Grocery",
            "region": "Midwest", "annual_payment_volume": 20_000_000.0,
            "visa_penetration": 0.65, "relationship_tier": "Developing",
            "last_outreach_date": as_of - pd.Timedelta(days=30),
            "outreach_count_ytd": 4, "contact_count": 3, "key_contact_active": True,
            "engagement_score": 60.0, "archetype": "steady",
            "debit_routing": True, "co_brand": False,
            "value_added_services": False, "acceptance_solutions": False,
        },
        {
            "merchant_id": "C", "merchant_name": "Gamma", "industry": "Travel",
            "region": "Northeast", "annual_payment_volume": 2_000_000.0,
            "visa_penetration": 0.95, "relationship_tier": "Light-Touch",
            "last_outreach_date": as_of - pd.Timedelta(days=10),
            "outreach_count_ytd": 9, "contact_count": 4, "key_contact_active": True,
            "engagement_score": 90.0, "archetype": "well_covered",
            "debit_routing": False, "co_brand": False,
            "value_added_services": False, "acceptance_solutions": False,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_opps():
    """Four hand-built opportunities with hand-computable weighted value."""
    rows = [
        # est_value, probability, stage
        ("OPP1", "A", "Acceptance", "Developing", 100_000.0, 0.50),
        ("OPP2", "A", "Co-brand", "Proposed", 200_000.0, 0.70),
        ("OPP3", "B", "Debit Routing", "Won", 50_000.0, 1.00),
        ("OPP4", "C", "VAS", "Lost", 40_000.0, 0.00),
    ]
    return pd.DataFrame(
        rows,
        columns=["opportunity_id", "merchant_id", "type", "stage",
                 "est_annual_value", "probability"],
    )


@pytest.fixture()
def empty_merchants():
    return pd.DataFrame(columns=[
        "merchant_id", "merchant_name", "industry", "region",
        "annual_payment_volume", "visa_penetration", "relationship_tier",
        "last_outreach_date", "outreach_count_ytd", "contact_count",
        "key_contact_active", "engagement_score", "archetype",
        "debit_routing", "co_brand", "value_added_services", "acceptance_solutions",
    ])


# ---------------------------------------------------------------------------
# 1. Primitive formulas - hand-computable
# ---------------------------------------------------------------------------

def test_coverage_rate_volume_weighted():
    # volume [100, 100], pen [0.4, 0.6] -> (40 + 60) / 200 = 0.50
    assert A.coverage_rate([100, 100], [0.4, 0.6]) == pytest.approx(0.50)


def test_coverage_rate_is_volume_weighted_not_simple_mean():
    # Big merchant at 0.2, tiny at 1.0 -> weighted pulls toward the big one.
    # (900*0.2 + 100*1.0) / 1000 = (180 + 100)/1000 = 0.28  (simple mean = 0.60)
    assert A.coverage_rate([900, 100], [0.2, 1.0]) == pytest.approx(0.28)


def test_coverage_rate_zero_volume_is_zero():
    assert A.coverage_rate([0, 0], [0.5, 0.5]) == 0.0


def test_whitespace_volume_elementwise():
    ws = A.whitespace_volume([100, 200], [0.25, 0.50])
    assert ws[0] == pytest.approx(75.0)   # 100 * (1 - 0.25)
    assert ws[1] == pytest.approx(100.0)  # 200 * (1 - 0.50)


def test_weighted_pipeline_value_hand_computed():
    # 100k*0.5 + 200k*0.7 = 50k + 140k = 190k
    assert A.weighted_pipeline_value([100_000, 200_000], [0.5, 0.7]) == pytest.approx(190_000.0)


# ---------------------------------------------------------------------------
# 2. prioritization_score on the hand-built tiny table
# ---------------------------------------------------------------------------

def test_prioritization_orders_whitespace_account_first(tiny_merchants):
    pri = A.prioritization_score(tiny_merchants)
    assert list(pri["merchant_id"]) == ["A", "B", "C"]
    assert pri.loc[0, "priority_rank"] == 1
    assert pri.loc[0, "merchant_id"] == "A"
    # Ranks are a clean 1..N with no gaps.
    assert list(pri["priority_rank"]) == [1, 2, 3]


def test_prioritization_components_hand_values(tiny_merchants):
    pri = A.prioritization_score(tiny_merchants).set_index("merchant_id")
    # A has max volume -> c_volume == 1.0; C has min -> 0.0
    assert pri.loc["A", "c_volume"] == pytest.approx(1.0)
    assert pri.loc["C", "c_volume"] == pytest.approx(0.0)
    # Opportunity component = open flags / 4.
    assert pri.loc["A", "c_opportunity"] == pytest.approx(4 / 4)
    assert pri.loc["B", "c_opportunity"] == pytest.approx(1 / 4)
    assert pri.loc["C", "c_opportunity"] == pytest.approx(0 / 4)
    # Whitespace dollars = volume * (1 - penetration).
    assert pri.loc["A", "whitespace_dollars"] == pytest.approx(100_000_000 * 0.70)
    # Score is bounded 0..100.
    assert (pri["priority_score"] >= 0).all() and (pri["priority_score"] <= 100).all()


def test_prioritization_score_is_monotonic_in_volume():
    """Holding everything else equal, more volume must not lower priority."""
    as_of = pd.Timestamp("2026-10-13")
    base = {
        "industry": "Retail", "region": "West", "visa_penetration": 0.5,
        "relationship_tier": "Core", "last_outreach_date": as_of - pd.Timedelta(days=50),
        "outreach_count_ytd": 3, "contact_count": 2, "key_contact_active": True,
        "engagement_score": 50.0, "archetype": "steady",
        "debit_routing": True, "co_brand": False,
        "value_added_services": False, "acceptance_solutions": False,
    }
    df = pd.DataFrame([
        {"merchant_id": "small", "merchant_name": "S", "annual_payment_volume": 10_000_000.0, **base},
        {"merchant_id": "big", "merchant_name": "B", "annual_payment_volume": 90_000_000.0, **base},
    ])
    pri = A.prioritization_score(df).set_index("merchant_id")
    assert pri.loc["big", "priority_score"] >= pri.loc["small", "priority_score"]


# ---------------------------------------------------------------------------
# 3. pipeline_summary / opportunity_breakdown - hand-computable
# ---------------------------------------------------------------------------

def test_pipeline_summary_weighting_and_open_filter(tiny_opps):
    pipe = A.pipeline_summary(tiny_opps)
    # Open = Developing + Proposed: 100k*0.5 + 200k*0.7 = 190k.
    assert pipe["total_weighted_open"] == pytest.approx(190_000.0)
    assert pipe["total_est_open"] == pytest.approx(300_000.0)  # 100k + 200k
    assert pipe["n_open"] == 2
    # Win rate = Won / (Won + Lost) = 1 / 2 = 0.5.
    assert pipe["win_rate"] == pytest.approx(0.5)


def test_pipeline_summary_by_stage_covers_all_present_stages(tiny_opps):
    pipe = A.pipeline_summary(tiny_opps)
    by_stage = pipe["by_stage"].set_index("stage")
    assert by_stage.loc["Won", "weighted_value"] == pytest.approx(50_000.0)  # prob 1.0
    assert by_stage.loc["Lost", "weighted_value"] == pytest.approx(0.0)      # prob 0.0
    assert by_stage.loc["Developing", "weighted_value"] == pytest.approx(50_000.0)


def test_opportunity_breakdown_open_only_excludes_terminal(tiny_opps):
    brk = A.opportunity_breakdown(tiny_opps, open_only=True)
    types = set(brk["type"])
    # Won (Debit Routing) and Lost (VAS) opps are terminal -> excluded.
    assert types == {"Acceptance", "Co-brand"}
    # Including all, Won/Lost reappear.
    brk_all = A.opportunity_breakdown(tiny_opps, open_only=False)
    assert set(brk_all["type"]) == {"Acceptance", "Co-brand", "Debit Routing", "VAS"}


# ---------------------------------------------------------------------------
# 4. coverage_gaps - hand-computable
# ---------------------------------------------------------------------------

def test_coverage_gaps_flags_stale_and_inactive(tiny_merchants):
    gaps = A.coverage_gaps(tiny_merchants, stale_days=90)
    # A is 200 days stale -> flagged; B (30d) and C (10d), both active -> not.
    assert set(gaps["merchant_id"]) == {"A"}
    row = gaps.iloc[0]
    assert row["stale_outreach"] is np.True_ or bool(row["stale_outreach"]) is True
    assert "no outreach" in row["gap_reason"]


def test_coverage_gaps_needs_contact_flag():
    as_of = pd.Timestamp("2026-10-13")
    df = pd.DataFrame([{
        "merchant_id": "X", "merchant_name": "X", "industry": "Retail", "region": "West",
        "annual_payment_volume": 5_000_000.0, "visa_penetration": 0.6,
        "relationship_tier": "Core",
        "last_outreach_date": as_of - pd.Timedelta(days=5),  # recent -> not stale
        "outreach_count_ytd": 8, "contact_count": 1, "key_contact_active": False,
        "engagement_score": 55.0, "archetype": "lapsed_contact",
        "debit_routing": False, "co_brand": False,
        "value_added_services": False, "acceptance_solutions": False,
    }])
    gaps = A.coverage_gaps(df, stale_days=90)
    # Recent outreach but inactive key contact -> still flagged, for replacement.
    assert len(gaps) == 1
    assert bool(gaps.iloc[0]["needs_contact"]) is True
    assert bool(gaps.iloc[0]["stale_outreach"]) is False
    assert "replace" in gaps.iloc[0]["gap_reason"]


# ---------------------------------------------------------------------------
# 5. Ground-truth validation against the full synthetic dataset
# ---------------------------------------------------------------------------

def test_prioritization_surfaces_injected_whitespace(merchants):
    pri = A.prioritization_score(merchants)
    # The top of the priority list should be dominated by injected whitespace.
    top20 = pri.head(20)
    assert (top20["archetype"] == "whitespace").mean() >= 0.75
    # Scores are sorted descending and ranks are 1..N.
    vals = pri["priority_score"].to_numpy()
    assert np.all(vals[:-1] >= vals[1:])
    assert list(pri["priority_rank"]) == list(range(1, len(pri) + 1))


def test_every_lapsed_contact_is_flagged_for_replacement(merchants):
    gaps = A.coverage_gaps(merchants)
    lapsed_ids = set(merchants.loc[merchants["archetype"] == "lapsed_contact", "merchant_id"])
    flagged_needs_contact = set(gaps.loc[gaps["needs_contact"], "merchant_id"])
    # Every injected lapsed-contact account must be surfaced for a replacement.
    assert lapsed_ids.issubset(flagged_needs_contact)
    # And all flagged accounts genuinely satisfy at least one gap condition.
    assert (gaps["stale_outreach"] | gaps["needs_contact"]).all()


def test_coverage_gaps_sorted_by_volume_desc(merchants):
    gaps = A.coverage_gaps(merchants)
    vals = gaps["annual_payment_volume"].to_numpy()
    assert np.all(vals[:-1] >= vals[1:])


def test_pipeline_weighting_matches_manual_sum(opportunities):
    pipe = A.pipeline_summary(opportunities)
    open_o = opportunities[opportunities["stage"].isin(OPEN_STAGES)]
    manual = float((open_o["est_annual_value"] * open_o["probability"]).sum())
    assert pipe["total_weighted_open"] == pytest.approx(manual, rel=1e-9)
    # Weighted value never exceeds raw estimated value.
    assert pipe["total_weighted_open"] <= pipe["total_est_open"] + 1e-6


def test_opportunity_types_consistent_with_eligibility(merchants, opportunities):
    """Each merchant's opportunities should map to an eligibility flag it has on
    (or be a Data Solutions play). This checks the generator is internally
    consistent - the pipeline reflects real whitespace, not noise."""
    flag_for_type = {
        "Debit Routing": "debit_routing",
        "Co-brand": "co_brand",
        "VAS": "value_added_services",
        "Acceptance": "acceptance_solutions",
    }
    m = merchants.set_index("merchant_id")
    bad = 0
    for o in opportunities.itertuples(index=False):
        if o.type == "Data Solutions":
            continue
        flag = flag_for_type[str(o.type)]
        if not bool(m.loc[o.merchant_id, flag]):
            bad += 1
    assert bad == 0


# ---------------------------------------------------------------------------
# 6. portfolio_summary totals
# ---------------------------------------------------------------------------

def test_portfolio_summary_totals(merchants, opportunities):
    summ = A.portfolio_summary(merchants, opportunities)
    assert summ["n_merchants"] == 300
    assert summ["total_volume"] == pytest.approx(float(merchants["annual_payment_volume"].sum()))
    # Coverage rate is volume-weighted and in (0, 1).
    assert 0.0 < summ["coverage_rate"] < 1.0
    # Whitespace volume = total * (1 - coverage) identity.
    implied = summ["total_volume"] * (1 - summ["coverage_rate"])
    assert summ["whitespace_volume"] == pytest.approx(implied, rel=1e-6)
    # by_industry rows sum back to the totals.
    assert summ["by_industry"]["volume"].sum() == pytest.approx(summ["total_volume"], rel=1e-9)
    assert int(summ["by_industry"]["merchants"].sum()) == summ["n_merchants"]


def test_portfolio_summary_open_opps_match_pipeline(merchants, opportunities):
    summ = A.portfolio_summary(merchants, opportunities)
    pipe = A.pipeline_summary(opportunities)
    assert summ["n_open_opportunities"] == pipe["n_open"]
    assert summ["weighted_pipeline"] == pytest.approx(pipe["total_weighted_open"], rel=1e-9)


# ---------------------------------------------------------------------------
# 7. Edge cases & reproducibility
# ---------------------------------------------------------------------------

def test_empty_inputs_do_not_crash(empty_merchants):
    summ = A.portfolio_summary(empty_merchants, None)
    assert summ["n_merchants"] == 0
    assert summ["total_volume"] == 0.0
    assert A.prioritization_score(empty_merchants).empty
    assert A.coverage_gaps(empty_merchants).empty
    empty_opps = pd.DataFrame(columns=[
        "opportunity_id", "merchant_id", "type", "stage",
        "est_annual_value", "probability"
    ])
    pipe = A.pipeline_summary(empty_opps)
    assert pipe["n_open"] == 0
    assert pipe["total_weighted_open"] == 0.0
    assert A.opportunity_breakdown(empty_opps).empty


def test_stage_probability_table_is_sane():
    # Terminal stages are exactly 0 and 1; open stages strictly between.
    assert STAGE_PROBABILITY["Won"] == 1.0
    assert STAGE_PROBABILITY["Lost"] == 0.0
    for s in OPEN_STAGES:
        assert 0.0 < STAGE_PROBABILITY[s] < 1.0


def test_reproducible_seed():
    a = generate_data(seed=7)
    b = generate_data(seed=7)
    assert a["merchants"].shape == b["merchants"].shape
    assert a["opportunities"].shape == b["opportunities"].shape
    assert a["merchants"]["annual_payment_volume"].sum() == b["merchants"]["annual_payment_volume"].sum()
