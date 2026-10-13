"""
data_gen.py
===========
Synthetic merchant-portfolio data generator for a Visa U.S. Merchant Business
Development analyst supporting a "Light Touch" merchant portfolio.

Business framing
----------------
A Merchant BD analyst owns a book of ~300 merchant relationships that are too
small for a dedicated account team but collectively material. The job is to read
portfolio data and decide: which accounts to prioritise, where the commercial
whitespace is (debit routing, co-brand, value-added services, acceptance
solutions, data solutions), which relationships have gone quiet, and which key
contacts have lapsed and need replacing - then keep an opportunity pipeline
moving against all of it.

This module fabricates two tables with deliberately injected patterns so the
analytics and tests have ground truth to validate against:

  * merchants    - one row per merchant relationship (volume, Visa penetration,
                   engagement, contact coverage, opportunity eligibility).
  * opportunities- one row per commercial opportunity in the pipeline (type,
                   stage, estimated annual value, probability, next step).

Injected merchant archetypes (ground truth, column ``archetype``):

  * whitespace    - HIGH volume, LOW Visa penetration, several open opportunity
                    flags, and STALE engagement. These are the accounts the
                    prioritisation model must surface at the top.
  * well_covered  - high penetration, active key contact, recent outreach,
                    little open whitespace. Low priority.
  * lapsed_contact- key contact has left / gone inactive and outreach is stale;
                    needs a contact-replacement play.
  * steady        - healthy middle-of-the-book relationships.

Everything is seeded for reproducibility. No real data is used anywhere.

Public API
----------
generate_data(seed=42) -> dict of DataFrames:
    {
      "merchants":     one row per merchant,
      "opportunities": one row per pipeline opportunity,
      "ground_truth":  the injected archetype per merchant (for test validation),
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_MERCHANTS = 300

# A fixed reference "today" so last_outreach_date math is reproducible and does
# not drift with the wall clock. Chosen to match the portfolio's as-of date.
REFERENCE_DATE = pd.Timestamp("2026-10-13")

# Merchant behavioural archetypes and their relative frequency.
ARCHETYPES = {
    "whitespace": 0.22,     # high volume, low penetration, open flags, stale -> PRIORITY
    "well_covered": 0.26,   # penetrated, active, recent -> low priority
    "lapsed_contact": 0.16,  # inactive key contact, stale outreach -> replace contact
    "steady": 0.36,         # healthy middle of the book
}

INDUSTRIES = [
    "Retail",
    "Grocery",
    "QSR/Restaurant",
    "Travel",
    "Fuel",
    "E-commerce",
    "Healthcare",
    "Entertainment",
]

REGIONS = ["West", "Southwest", "Midwest", "Southeast", "Northeast"]

RELATIONSHIP_TIERS = ["Strategic", "Core", "Developing", "Light-Touch"]

# Opportunity types map to the four eligibility flags (+ a data play).
OPP_TYPES = ["Debit Routing", "Co-brand", "VAS", "Acceptance", "Data Solutions"]

# Pipeline stages and the win probability we attach to each. Won/Lost are
# terminal; the rest are "open" pipeline.
STAGE_PROBABILITY = {
    "Identified": 0.10,
    "Qualified": 0.25,
    "Developing": 0.45,
    "Proposed": 0.70,
    "Won": 1.00,
    "Lost": 0.00,
}

# Plausible brand-like but clearly invented merchant name parts.
_NAME_PREFIX = [
    "Summit", "Harbor", "Cedar", "Vertex", "Lumen", "Pioneer", "Atlas",
    "Beacon", "Granite", "Meridian", "Copper", "Sterling", "Cascade",
    "Northwind", "Evergreen", "Ironwood", "Bluewater", "Redwood", "Keystone",
    "Brightline", "Clearfield", "Fairview", "Highland", "Lakeside",
]
_NAME_SUFFIX = [
    "Markets", "Group", "Holdings", "Retail", "Brands", "Partners", "Supply",
    "Fuels", "Hospitality", "Commerce", "Goods", "Trading", "Stores",
    "Outfitters", "Collective", "Provisions", "Logistics", "Ventures",
]


# ---------------------------------------------------------------------------
# Merchant table
# ---------------------------------------------------------------------------

def _build_merchant_names(rng: np.random.Generator, n: int) -> list[str]:
    """Invent clearly-fake but brand-like merchant names, de-duplicated."""
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        pre = rng.choice(_NAME_PREFIX)
        suf = rng.choice(_NAME_SUFFIX)
        base = f"{pre} {suf}"
        name = base if base not in seen else f"{base} {len(names):03d}"
        seen.add(name)
        names.append(name)
    return names


def _build_merchants(rng: np.random.Generator) -> pd.DataFrame:
    """Build the ~300-merchant portfolio with injected archetypes."""
    ids = np.arange(1, N_MERCHANTS + 1)
    merchant_id = [f"M{ i:04d}".replace(" ", "") for i in ids]

    arch_names = list(ARCHETYPES.keys())
    arch_probs = np.array(list(ARCHETYPES.values()))
    arch_probs = arch_probs / arch_probs.sum()
    archetype = rng.choice(arch_names, size=N_MERCHANTS, p=arch_probs)

    industry = rng.choice(INDUSTRIES, size=N_MERCHANTS)
    region = rng.choice(REGIONS, size=N_MERCHANTS)

    # --- Annual payment volume ($) -----------------------------------------
    # Log-normal spread so a few large merchants dominate volume (realistic).
    # Whitespace accounts skew large (that is what makes them worth the chase).
    base_vol = rng.lognormal(mean=16.1, sigma=0.9, size=N_MERCHANTS)  # ~ $10M median
    vol_mult = np.select(
        [archetype == "whitespace", archetype == "well_covered"],
        [1.9, 1.1],
        default=1.0,
    )
    annual_payment_volume = np.round(np.clip(base_vol * vol_mult, 5e5, 9e8), -3)

    # --- Visa penetration (share of their card volume already on Visa) -----
    penetration = np.select(
        [
            archetype == "whitespace",     # lots of room to grow
            archetype == "well_covered",   # already deep
            archetype == "lapsed_contact",
        ],
        [
            rng.uniform(0.28, 0.55, N_MERCHANTS),
            rng.uniform(0.84, 0.98, N_MERCHANTS),
            rng.uniform(0.45, 0.72, N_MERCHANTS),
        ],
        default=rng.uniform(0.55, 0.82, N_MERCHANTS),
    )
    visa_penetration = np.round(penetration, 3)

    # --- Relationship tier --------------------------------------------------
    # Bigger / stickier accounts sit in higher tiers.
    tier = np.where(
        annual_payment_volume > 1.5e8, "Strategic",
        np.where(
            annual_payment_volume > 5e7, "Core",
            np.where(annual_payment_volume > 1.5e7, "Developing", "Light-Touch"),
        ),
    )

    # --- Engagement / outreach ---------------------------------------------
    # Stale archetypes have low engagement and old last-outreach dates.
    engagement_score = np.select(
        [
            archetype == "well_covered",
            archetype == "whitespace",
            archetype == "lapsed_contact",
        ],
        [
            rng.uniform(70, 95, N_MERCHANTS),
            rng.uniform(15, 45, N_MERCHANTS),
            rng.uniform(10, 38, N_MERCHANTS),
        ],
        default=rng.uniform(40, 72, N_MERCHANTS),
    ).round(1)

    days_since_outreach = np.select(
        [
            archetype == "well_covered",
            archetype == "whitespace",
            archetype == "lapsed_contact",
        ],
        [
            rng.integers(3, 45, N_MERCHANTS),
            rng.integers(95, 260, N_MERCHANTS),
            rng.integers(130, 330, N_MERCHANTS),
        ],
        default=rng.integers(20, 110, N_MERCHANTS),
    ).astype(int)
    last_outreach_date = REFERENCE_DATE - pd.to_timedelta(days_since_outreach, unit="D")

    outreach_count_ytd = np.select(
        [
            archetype == "well_covered",
            archetype == "whitespace",
            archetype == "lapsed_contact",
        ],
        [
            rng.integers(6, 14, N_MERCHANTS),
            rng.integers(0, 3, N_MERCHANTS),
            rng.integers(0, 2, N_MERCHANTS),
        ],
        default=rng.integers(2, 7, N_MERCHANTS),
    ).astype(int)

    # --- Contact coverage ---------------------------------------------------
    contact_count = np.clip(
        rng.poisson(lam=np.where(archetype == "well_covered", 4, 2), size=N_MERCHANTS),
        1, 9,
    ).astype(int)
    # Key contact active? Lapsed-contact accounts always have an inactive key
    # contact; well-covered almost always active; others mostly active.
    key_active_prob = np.select(
        [archetype == "lapsed_contact", archetype == "well_covered"],
        [0.0, 0.97],
        default=0.82,
    )
    key_contact_active = rng.random(N_MERCHANTS) < key_active_prob

    df = pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "merchant_name": _build_merchant_names(rng, N_MERCHANTS),
            "industry": industry,
            "region": region,
            "annual_payment_volume": annual_payment_volume,
            "visa_penetration": visa_penetration,
            "relationship_tier": tier,
            "last_outreach_date": last_outreach_date,
            "outreach_count_ytd": outreach_count_ytd,
            "contact_count": contact_count,
            "key_contact_active": key_contact_active,
            "engagement_score": engagement_score,
            "archetype": archetype,
        }
    )

    # --- Opportunity eligibility flags -------------------------------------
    # Whitespace accounts light up several flags; well-covered accounts few.
    # Each flag probability also nudged by industry fit where it makes sense.
    def _flag(base_prob_by_arch: dict[str, float], extra: np.ndarray | None = None):
        p = df["archetype"].map(base_prob_by_arch).to_numpy(dtype=float)
        if extra is not None:
            p = np.clip(p + extra, 0.0, 0.98)
        return rng.random(N_MERCHANTS) < p

    debit_boost = np.where(df["industry"].isin(["Grocery", "Fuel", "QSR/Restaurant"]), 0.15, 0.0)
    cobrand_boost = np.where(df["industry"].isin(["Retail", "Travel", "E-commerce"]), 0.15, 0.0)
    accept_boost = np.where(df["industry"].isin(["Healthcare", "Entertainment", "QSR/Restaurant"]), 0.12, 0.0)

    df["debit_routing"] = _flag(
        {"whitespace": 0.80, "steady": 0.45, "lapsed_contact": 0.50, "well_covered": 0.12},
        debit_boost,
    )
    df["co_brand"] = _flag(
        {"whitespace": 0.70, "steady": 0.35, "lapsed_contact": 0.40, "well_covered": 0.10},
        cobrand_boost,
    )
    df["value_added_services"] = _flag(
        {"whitespace": 0.78, "steady": 0.42, "lapsed_contact": 0.48, "well_covered": 0.15},
    )
    df["acceptance_solutions"] = _flag(
        {"whitespace": 0.72, "steady": 0.40, "lapsed_contact": 0.45, "well_covered": 0.12},
        accept_boost,
    )

    return df


# ---------------------------------------------------------------------------
# Opportunity (pipeline) table
# ---------------------------------------------------------------------------

_FLAG_TO_TYPE = {
    "debit_routing": "Debit Routing",
    "co_brand": "Co-brand",
    "value_added_services": "VAS",
    "acceptance_solutions": "Acceptance",
}

_NEXT_STEPS = [
    "Schedule discovery call",
    "Send pricing proposal",
    "Share VAS product overview",
    "Loop in Visa solutions engineer",
    "Confirm economics with merchant finance",
    "Draft co-brand term sheet",
    "Book executive business review",
    "Awaiting merchant legal review",
    "Re-engage new key contact",
]


def _build_opportunities(rng: np.random.Generator, merchants: pd.DataFrame) -> pd.DataFrame:
    """Create a pipeline of opportunities keyed to merchant eligibility flags.

    An opportunity's TYPE is drawn from the merchant's active eligibility flags
    (plus an occasional Data Solutions play), so the pipeline is consistent with
    the portfolio's whitespace. Whitespace merchants spawn more opportunities;
    well-covered merchants spawn few. Stage distribution skews early (most
    opportunities are freshly identified), with a realistic Won/Lost tail.
    """
    rows = []
    opp_counter = 1
    stages = list(STAGE_PROBABILITY.keys())
    # Early-weighted stage mix.
    stage_probs = np.array([0.34, 0.24, 0.17, 0.11, 0.08, 0.06])
    stage_probs = stage_probs / stage_probs.sum()

    for m in merchants.itertuples(index=False):
        active_flags = [
            _FLAG_TO_TYPE[f]
            for f in ("debit_routing", "co_brand", "value_added_services", "acceptance_solutions")
            if getattr(m, f)
        ]
        # Occasionally add a Data Solutions play for engaged, larger merchants.
        if m.annual_payment_volume > 2e7 and rng.random() < 0.25:
            active_flags.append("Data Solutions")

        if not active_flags:
            continue

        # How many opportunities this merchant carries.
        if m.archetype == "whitespace":
            n_opp = int(rng.integers(2, 4))
        elif m.archetype == "well_covered":
            n_opp = int(rng.integers(0, 2))
        else:
            n_opp = int(rng.integers(1, 3))
        n_opp = min(n_opp, len(active_flags))
        if n_opp <= 0:
            continue

        chosen_types = rng.choice(active_flags, size=n_opp, replace=False)

        for opp_type in chosen_types:
            stage = rng.choice(stages, p=stage_probs)
            # Estimated annual value scales with the merchant's volume and the
            # share of volume a given solution could touch.
            type_share = {
                "Debit Routing": 0.012,
                "Co-brand": 0.020,
                "VAS": 0.008,
                "Acceptance": 0.015,
                "Data Solutions": 0.006,
            }[str(opp_type)]
            noise = rng.uniform(0.6, 1.5)
            est_annual_value = round(m.annual_payment_volume * type_share * noise, -2)
            est_annual_value = float(max(5_000.0, est_annual_value))

            # Probability from the stage, with a little per-deal jitter but kept
            # exactly 0/1 for the terminal Lost/Won stages.
            base_p = STAGE_PROBABILITY[str(stage)]
            if stage in ("Won", "Lost"):
                probability = base_p
            else:
                probability = float(np.clip(base_p + rng.uniform(-0.05, 0.05), 0.02, 0.95))
                probability = round(probability, 3)

            created_offset = int(rng.integers(10, 300))
            created_date = REFERENCE_DATE - pd.Timedelta(days=created_offset)

            rows.append(
                {
                    "opportunity_id": f"OPP{opp_counter:05d}",
                    "merchant_id": m.merchant_id,
                    "type": str(opp_type),
                    "stage": str(stage),
                    "est_annual_value": est_annual_value,
                    "probability": probability,
                    "next_step": rng.choice(_NEXT_STEPS),
                    "created_date": created_date,
                }
            )
            opp_counter += 1

    opps = pd.DataFrame(rows)
    return opps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_data(seed: int = 42) -> dict:
    """Generate the full synthetic merchant-portfolio dataset.

    Returns a dict with keys ``merchants``, ``opportunities``, ``ground_truth``.
    """
    rng = np.random.default_rng(seed)

    merchants = _build_merchants(rng)
    opportunities = _build_opportunities(rng, merchants)

    # Light dtype tidy-up for repeated string columns.
    for col in ("industry", "region", "relationship_tier", "archetype"):
        merchants[col] = merchants[col].astype("category")
    if not opportunities.empty:
        for col in ("type", "stage"):
            opportunities[col] = opportunities[col].astype("category")

    ground_truth = merchants[["merchant_id", "archetype"]].copy()

    return {
        "merchants": merchants,
        "opportunities": opportunities,
        "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    data = generate_data()
    m = data["merchants"]
    o = data["opportunities"]
    print("merchants:", m.shape)
    print("opportunities:", o.shape)
    print("\nArchetype mix:")
    print(m["archetype"].value_counts())
    print("\nTotal portfolio volume $:", round(m["annual_payment_volume"].sum(), 2))
    print("Avg Visa penetration:", round(m["visa_penetration"].mean(), 3))
    print("\nOpportunity type mix:")
    print(o["type"].value_counts())
    print("\nStage mix:")
    print(o["stage"].value_counts())
    print("\nWeighted pipeline $:", round((o["est_annual_value"] * o["probability"]).sum(), 2))
