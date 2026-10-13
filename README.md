# Merchant Portfolio & Opportunity Analytics

A prioritisation, pipeline, and coverage dashboard for a **Visa U.S. Merchant
Business Development analyst** supporting a "Light Touch" merchant portfolio
(~300 relationships). It turns raw portfolio data into the deliverables a BD
analyst actually produces: **which accounts to work first, where the commercial
whitespace is, what is in the opportunity pipeline, and which relationships have
gone quiet or lost their key contact.**

> **Honesty note:** all data here is **synthetic** - generated on launch with a
> fixed random seed. No real Visa or merchant data is used. This is a portfolio
> demonstration of analytical approach, metric design, and merchant-portfolio
> judgment, not a report on any real business.

**Live app:** _(to be added)_

---

## How it maps to the Visa Merchant BD Analyst role

The "Analyst, U.S. Merchant Business Development" role supports the Light Touch
Merchant Portfolio: account prioritisation, identifying commercial opportunities
(debit routing, co-brand, value-added services, acceptance solutions, data
solutions), maintaining an opportunity pipeline, tracking engagement / outreach
and contact coverage, and turning portfolio data into prioritisation
recommendations. This dashboard produces exactly those deliverables, end to end.

## The views

1. **Portfolio overview** - headline KPIs (merchants, total volume, avg Visa
   penetration, open opportunities, weighted pipeline), plus volume-by-industry
   and the penetration distribution that exposes the growth whitespace.
2. **Account prioritization** - a transparent 0-100 score ranking merchants by
   commercial whitespace (high volume + LOW penetration + open opportunity flags
   + stale engagement), with a volume-vs-penetration scatter that highlights the
   priority accounts and a downloadable ranked call list.
3. **Opportunity pipeline** - a probability-weighted funnel by stage, weighted
   value by solution type, and a next-steps list on the highest-value deals.
4. **Engagement & coverage** - outreach-recency distribution, the biggest
   relationships that have gone quiet (re-engage), and accounts whose key contact
   is inactive (replace the contact).

The visual design follows Cole Nussbaumer Knaflic's *Storytelling with Data*:
takeaway titles (the chart title states the insight), one restrained accent color
plus a reserved status palette, decluttered charts, direct labels, and a "so what"
on every view.

## Architecture

| File | Role |
|------|------|
| `data_gen.py` | Synthetic data generator. ~300 merchants with volume, Visa penetration, engagement, contact coverage, and opportunity-eligibility flags, plus a linked opportunity pipeline. Injected behavioral archetypes (whitespace / well-covered / lapsed-contact / steady) give the analytics ground truth to validate against. Seeded and reproducible. |
| `analytics.py` | Pure pandas/numpy analytics - **no UI imports.** `portfolio_summary`, `prioritization_score`, `pipeline_summary`, `opportunity_breakdown`, `coverage_gaps`, and the primitive formulas behind them. Each function documents the business question it answers. |
| `test_analytics.py` | `pytest` suite: hand-computable fixtures for the metric formulas plus ground-truth checks (prioritisation surfaces whitespace, every lapsed contact is flagged, pipeline weighting math is consistent). |
| `app.py` | The Streamlit multi-view dashboard. |

The separation matters: `analytics.py` is testable in isolation and reusable from
a notebook or batch job; the app is a thin presentation layer over it.

## Key metrics

- **Visa penetration** = share of a merchant's card volume already running on Visa.
- **Whitespace volume** = volume x (1 - penetration), the un-penetrated dollars.
- **Coverage rate** = volume-weighted share of portfolio volume on Visa.
- **Priority score** = weighted blend of volume, whitespace, open opportunity
  flags, and engagement gap (0-100).
- **Weighted pipeline** = estimated annual value x stage win probability.

## The priority logic

| Component | Weight | Signal |
|---|---|---|
| **Volume** | 35% | Annual payment volume (bigger book = more at stake) |
| **Whitespace** | 30% | Volume x (1 - penetration) - the un-penetrated dollars |
| **Opportunity** | 20% | Share of the 4 eligibility flags that are open |
| **Engagement gap** | 15% | Stale outreach + low engagement = being neglected |

## Running locally

```bash
pip install -r requirements.txt

# Run the test suite
python -m pytest -q

# Launch the dashboard
streamlit run app.py
```

The dataset is generated in-process on first launch and cached for the session,
so there is no data file to download or manage.

## Tech

Python, pandas, numpy, Streamlit, Plotly, pytest.
