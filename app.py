"""
app.py
======
Merchant Portfolio & Opportunity Analytics - a Streamlit dashboard for a Visa
U.S. Merchant Business Development analyst supporting a "Light Touch" merchant
portfolio (~300 relationships).

The app turns a synthetic merchant portfolio into the four things a Merchant BD
analyst actually produces:

  1. Portfolio Overview       - the headline KPIs and where the volume / whitespace sit.
  2. Account Prioritization   - a ranked call list of high-volume, low-penetration whitespace.
  3. Opportunity Pipeline     - a probability-weighted funnel by stage and solution type.
  4. Engagement & Coverage    - stale accounts and lapsed key contacts to re-engage / replace.

Design follows Cole Nussbaumer Knaflic's "Storytelling with Data": every view
leads with a takeaway title (the insight, not the metric), color is restrained to
one accent plus a reserved status palette, charts are decluttered, and each view
ends with a "so what". All data is synthetic and generated on launch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from data_gen import generate_data

# ---------------------------------------------------------------------------
# Palette (validated, colorblind-safe; single accent + muted context)
# ---------------------------------------------------------------------------
ACCENT = "#1a1f71"        # Visa-ish navy - the one accent, used only on the point
ACCENT2 = "#f7b600"       # secondary highlight (Visa gold) for a second emphasis
CONTEXT = "#c3c2b7"       # muted gray for everything that is context
INK = "#52514e"           # secondary ink for labels
GRID = "#e1e0d9"          # hairline gridlines
STATUS = {
    "priority": "#1a1f71",   # navy - act now
    "warn": "#f7b600",       # gold - watch
    "risk": "#d03b3b",       # red - at risk
    "ok": CONTEXT,           # context - no action
}

st.set_page_config(
    page_title="Merchant Portfolio & Opportunity Analytics",
    page_icon=":material/hub:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data + analytics (generated and computed once, then cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating synthetic merchant-portfolio data...")
def load_everything(seed: int = 42) -> dict:
    """Generate the synthetic dataset and pre-compute the analytics layer once."""
    data = generate_data(seed=seed)
    merchants = data["merchants"]
    opps = data["opportunities"]
    return {
        "merchants": merchants,
        "opportunities": opps,
        "summary": A.portfolio_summary(merchants, opps),
        "priority": A.prioritization_score(merchants),
        "pipeline": A.pipeline_summary(opps),
        "breakdown": A.opportunity_breakdown(opps),
        "gaps": A.coverage_gaps(merchants),
    }


BUNDLE = load_everything()


# ---------------------------------------------------------------------------
# Small formatting / charting helpers
# ---------------------------------------------------------------------------

def money(x: float, decimals: int = 0) -> str:
    """Compact dollar formatting: $1.2B, $340M, $85K."""
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x / 1_000_000_000:.1f}B"
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.0f}M"
    if x >= 1_000:
        return f"{sign}${x / 1_000:.0f}K"
    return f"{sign}${x:,.{decimals}f}"


def _declutter(fig: go.Figure, height: int = 380) -> go.Figure:
    """Strip chartjunk the Knaflic way: no top/right spines, hairline grid, muted ink."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        title=dict(font=dict(size=17, color="#0b0b0b"), x=0.0, xanchor="left"),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, ticks="")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="")
    return fig


# ---------------------------------------------------------------------------
# View 1 - Portfolio Overview
# ---------------------------------------------------------------------------

def view_overview(bundle: dict) -> None:
    s = bundle["summary"]
    merchants = bundle["merchants"]

    st.subheader("Portfolio overview")
    st.caption(
        "A ~300-merchant 'Light Touch' book across 8 industries and 5 regions. "
        "Synthetic, seeded data generated on launch."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Merchants", f"{s['n_merchants']:,}")
    c2.metric("Portfolio volume", money(s["total_volume"]))
    c3.metric("Avg Visa penetration", f"{s['avg_visa_penetration']:.0%}")
    c4.metric("Open opportunities", f"{s['n_open_opportunities']:,}")
    c5.metric("Weighted pipeline", money(s["weighted_pipeline"]))

    st.divider()

    # --- Volume & whitespace by industry -----------------------------------
    ind = s["by_industry"].sort_values("volume")
    top_ind = ind.loc[ind["volume"].idxmax(), "industry"]
    colors = [ACCENT if i == top_ind else CONTEXT for i in ind["industry"]]

    fig = go.Figure(
        go.Bar(
            x=ind["volume"], y=ind["industry"], orientation="h",
            marker_color=colors,
            text=[money(v) for v in ind["volume"]],
            textposition="outside", textfont=dict(color=INK, size=12),
            hovertemplate="%{y}<br>Volume: %{x:,.0f}<extra></extra>",
        )
    )
    _declutter(fig, height=420)
    fig.update_layout(
        title=f"{top_ind} is the largest slice of portfolio payment volume",
        xaxis_title=None, yaxis_title=None,
    )
    fig.update_xaxes(showticklabels=False, range=[0, ind["volume"].max() * 1.18])
    st.plotly_chart(fig, width="stretch")

    # --- Penetration distribution ------------------------------------------
    st.markdown("##### Most of the book is only partly on Visa - the growth whitespace")
    pen = merchants["visa_penetration"]
    median_pen = float(pen.median())
    below = int((pen < 0.60).sum())
    counts, edges = np.histogram(pen, bins=np.linspace(0, 1, 21))
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [ACCENT if c < 0.60 else CONTEXT for c in centers]

    fig2 = go.Figure(
        go.Bar(
            x=centers, y=counts, width=0.045,
            marker_color=bar_colors,
            hovertemplate="Penetration ~%{x:.0%}<br>Merchants: %{y}<extra></extra>",
        )
    )
    fig2.add_vline(x=median_pen, line_width=1, line_dash="dot", line_color=INK)
    _declutter(fig2, height=380)
    fig2.update_layout(
        title=f"{below} merchants sit below 60% Visa penetration - room to grow share",
    )
    fig2.update_xaxes(title="Visa penetration", tickformat=".0%", showgrid=False)
    fig2.update_yaxes(title="Merchants")
    st.plotly_chart(fig2, width="stretch")

    st.info(
        f"**So what:** the portfolio does **{money(s['total_volume'])}** in annual payment "
        f"volume but only **{s['coverage_rate']:.0%}** of it runs on Visa today - leaving "
        f"**{money(s['whitespace_volume'])}** of un-penetrated whitespace. The navy bars above "
        f"(below-median penetration) are where the acquisition dollars are. The next view ranks "
        f"exactly which accounts to work first.",
        icon=":material/lightbulb:",
    )


# ---------------------------------------------------------------------------
# View 2 - Account Prioritization
# ---------------------------------------------------------------------------

def view_prioritization(bundle: dict) -> None:
    pri = bundle["priority"]

    st.subheader("Account prioritization")
    st.caption(
        "A transparent 0-100 score: high volume + LOW Visa penetration + open "
        "opportunity flags + stale engagement = higher priority."
    )

    # Define the whitespace priority set: top quartile of score.
    cutoff = float(pri["priority_score"].quantile(0.75))
    priority_set = pri[pri["priority_score"] >= cutoff]
    n_priority = len(priority_set)
    priority_ws = float(priority_set["whitespace_dollars"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Priority accounts", f"{n_priority}", "top quartile by score")
    c2.metric("Whitespace in priority set", money(priority_ws))
    c3.metric("Avg penetration (priority)", f"{priority_set['visa_penetration'].mean():.0%}",
              f"vs {pri['visa_penetration'].mean():.0%} book", delta_color="off")

    st.divider()

    # --- Scatter: volume vs penetration, bubble = priority score -----------
    st.markdown("##### The target zone: big books that are still light on Visa")
    is_priority = pri["priority_score"] >= cutoff
    pt_colors = np.where(is_priority, ACCENT, CONTEXT)
    sizes = np.clip(pri["priority_score"] / pri["priority_score"].max() * 34, 6, 34)

    fig = go.Figure(
        go.Scatter(
            x=pri["annual_payment_volume"], y=pri["visa_penetration"],
            mode="markers",
            marker=dict(size=sizes, color=pt_colors, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            text=pri["merchant_name"],
            customdata=np.stack([pri["priority_score"], pri["open_opportunities"]], axis=-1),
            hovertemplate=(
                "%{text}<br>Volume: %{x:,.0f}<br>Penetration: %{y:.0%}"
                "<br>Score: %{customdata[0]:.0f}  ·  Open flags: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.60, line_width=1, line_dash="dot", line_color=INK)
    _declutter(fig, height=440)
    fig.update_layout(
        title=f"{n_priority} high-volume, low-penetration accounts are the priority (navy)",
    )
    fig.update_xaxes(title="Annual payment volume ($)", showgrid=False)
    fig.update_yaxes(title="Visa penetration", tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    # --- Ranked table ------------------------------------------------------
    st.markdown("##### Ranked call list")
    top_n = st.slider("Show top N accounts", 10, 100, 25, step=5)
    show = pri.head(top_n)[[
        "priority_rank", "merchant_id", "merchant_name", "industry", "relationship_tier",
        "annual_payment_volume", "visa_penetration", "whitespace_dollars",
        "open_opportunities", "days_since_outreach", "priority_score",
    ]].rename(columns={
        "priority_rank": "#", "merchant_id": "ID", "merchant_name": "Merchant",
        "industry": "Industry", "relationship_tier": "Tier",
        "annual_payment_volume": "Volume", "visa_penetration": "Penetration",
        "whitespace_dollars": "Whitespace $", "open_opportunities": "Open flags",
        "days_since_outreach": "Days since outreach", "priority_score": "Score",
    })
    show["Penetration"] = (show["Penetration"] * 100).round(0)
    st.dataframe(
        show, width="stretch", hide_index=True,
        column_config={
            "Volume": st.column_config.NumberColumn(format="$%.0f"),
            "Penetration": st.column_config.NumberColumn(format="%.0f%%"),
            "Whitespace $": st.column_config.NumberColumn(format="$%.0f",
                help="Volume not yet on Visa = volume x (1 - penetration)"),
            "Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    csv = pri.to_csv(index=False).encode("utf-8")
    st.download_button("Download full ranked list (CSV)", data=csv,
                       file_name="account_prioritization.csv", mime="text/csv")

    st.info(
        "**So what:** this is the analyst's call list. The navy accounts in the "
        "lower-right - large volume, still under 60% Visa penetration - are where a "
        "single conversation moves the most volume. Each carries open opportunity "
        "flags that name the specific play (debit routing, co-brand, VAS, acceptance).",
        icon=":material/lightbulb:",
    )


# ---------------------------------------------------------------------------
# View 3 - Opportunity Pipeline
# ---------------------------------------------------------------------------

def view_pipeline(bundle: dict) -> None:
    pipe = bundle["pipeline"]
    opps = bundle["opportunities"]
    merchants = bundle["merchants"]

    st.subheader("Opportunity pipeline")
    st.caption("Probability-weighted value across the funnel, by stage and solution type.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Open opportunities", f"{pipe['n_open']:,}")
    c2.metric("Weighted open pipeline", money(pipe["total_weighted_open"]),
              f"of {money(pipe['total_est_open'])} raw", delta_color="off")
    c3.metric("Win rate (closed)", f"{pipe['win_rate']:.0%}")

    st.divider()

    # --- Funnel by stage (open stages) -------------------------------------
    stage_order = ["Identified", "Qualified", "Developing", "Proposed"]
    bs = pipe["by_stage"].set_index("stage").reindex(stage_order).reset_index()
    bs = bs.dropna(subset=["weighted_value"])
    fig = go.Figure(
        go.Funnel(
            y=bs["stage"], x=bs["weighted_value"],
            textinfo="value",
            texttemplate="%{value:$,.0f}",
            marker=dict(color=[CONTEXT, CONTEXT, ACCENT2, ACCENT]),
            connector=dict(line=dict(color=GRID)),
            hovertemplate="%{y}<br>Weighted $: %{x:,.0f}<extra></extra>",
        )
    )
    _declutter(fig, height=380)
    late = float(bs[bs["stage"].isin(["Developing", "Proposed"])]["weighted_value"].sum())
    fig.update_layout(
        title=f"{money(late)} of weighted pipeline is in late stages (Developing + Proposed)",
    )
    st.plotly_chart(fig, width="stretch")

    # --- Weighted value by opportunity type --------------------------------
    st.markdown("##### Where the weighted pipeline concentrates by solution")
    brk = pipe["by_type"].sort_values("weighted_value")
    top_type = brk.loc[brk["weighted_value"].idxmax(), "type"]
    colors = [ACCENT if t == top_type else CONTEXT for t in brk["type"]]
    fig2 = go.Figure(
        go.Bar(
            x=brk["weighted_value"], y=brk["type"], orientation="h",
            marker_color=colors,
            text=[money(v) for v in brk["weighted_value"]],
            textposition="outside", textfont=dict(color=INK, size=12),
            hovertemplate="%{y}<br>Weighted $: %{x:,.0f}<extra></extra>",
        )
    )
    _declutter(fig2, height=360)
    fig2.update_layout(title=f"{top_type} carries the largest weighted opportunity value")
    fig2.update_xaxes(showticklabels=False, range=[0, brk["weighted_value"].max() * 1.2])
    st.plotly_chart(fig2, width="stretch")

    # --- Next-steps list (late-stage, highest weighted value first) --------
    st.markdown("##### Next steps on the highest-value open opportunities")
    o = opps[opps["stage"].isin(stage_order)].copy()
    o["weighted_value"] = o["est_annual_value"] * o["probability"]
    o = o.merge(merchants[["merchant_id", "merchant_name"]], on="merchant_id", how="left")
    o = o.sort_values("weighted_value", ascending=False).head(20)
    show = o[[
        "opportunity_id", "merchant_name", "type", "stage",
        "est_annual_value", "probability", "weighted_value", "next_step",
    ]].rename(columns={
        "opportunity_id": "Opp", "merchant_name": "Merchant", "type": "Type",
        "stage": "Stage", "est_annual_value": "Est value", "probability": "Prob",
        "weighted_value": "Weighted $", "next_step": "Next step",
    })
    st.dataframe(
        show, width="stretch", hide_index=True,
        column_config={
            "Est value": st.column_config.NumberColumn(format="$%.0f"),
            "Prob": st.column_config.NumberColumn(format="%.0f%%"),
            "Weighted $": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    st.info(
        "**So what:** the weighted view keeps the pipeline honest - a wall of "
        "freshly-identified leads is discounted to what it is really worth, while "
        "late-stage deals carry their full probability. The next-steps list turns "
        "that into concrete actions for the week.",
        icon=":material/lightbulb:",
    )


# ---------------------------------------------------------------------------
# View 4 - Engagement & Coverage
# ---------------------------------------------------------------------------

def view_coverage(bundle: dict) -> None:
    gaps = bundle["gaps"]
    merchants = bundle["merchants"]

    st.subheader("Engagement & coverage")
    st.caption("Relationships that have gone quiet, and key contacts that need replacing.")

    stale = gaps[gaps["stale_outreach"]]
    needs_contact = gaps[gaps["needs_contact"]]
    at_risk_vol = float(gaps["annual_payment_volume"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts flagged", f"{len(gaps)}", f"{len(gaps)/len(merchants):.0%} of book",
              delta_color="off")
    c2.metric("Stale outreach (90d+)", f"{len(stale)}")
    c3.metric("Key contact to replace", f"{len(needs_contact)}")

    st.divider()

    # --- Outreach recency distribution -------------------------------------
    st.markdown("##### Outreach recency across the book")
    days = (
        (pd.Timestamp(merchants["last_outreach_date"].max())
         - pd.to_datetime(merchants["last_outreach_date"])).dt.days
    )
    counts, edges = np.histogram(days, bins=np.linspace(0, max(300, days.max()), 16))
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [STATUS["risk"] if c > 90 else CONTEXT for c in centers]
    fig = go.Figure(
        go.Bar(
            x=centers, y=counts, width=(edges[1] - edges[0]) * 0.9,
            marker_color=bar_colors,
            hovertemplate="~%{x:.0f} days ago<br>Merchants: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=90, line_width=1, line_dash="dot", line_color=INK)
    _declutter(fig, height=360)
    fig.update_layout(
        title=f"{len(stale)} accounts (red) have had no outreach in over 90 days",
    )
    fig.update_xaxes(title="Days since last outreach", showgrid=False)
    fig.update_yaxes(title="Merchants")
    st.plotly_chart(fig, width="stretch")

    # --- Re-engage: stale, highest-volume accounts -------------------------
    st.markdown("##### Re-engage first: biggest relationships that have gone quiet")
    re = stale.sort_values("annual_payment_volume", ascending=False).head(15)
    show_re = re[[
        "merchant_id", "merchant_name", "industry", "annual_payment_volume",
        "visa_penetration", "days_since_outreach", "engagement_score",
    ]].rename(columns={
        "merchant_id": "ID", "merchant_name": "Merchant", "industry": "Industry",
        "annual_payment_volume": "Volume", "visa_penetration": "Penetration",
        "days_since_outreach": "Days since outreach", "engagement_score": "Engagement",
    })
    show_re["Penetration"] = (show_re["Penetration"] * 100).round(0)
    st.dataframe(
        show_re, width="stretch", hide_index=True,
        column_config={
            "Volume": st.column_config.NumberColumn(format="$%.0f"),
            "Penetration": st.column_config.NumberColumn(format="%.0f%%"),
            "Engagement": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # --- Replace contact: inactive key contacts ----------------------------
    st.markdown("##### Replace the key contact: relationships with no active owner")
    rc = needs_contact.sort_values("annual_payment_volume", ascending=False).head(15)
    show_rc = rc[[
        "merchant_id", "merchant_name", "industry", "annual_payment_volume",
        "contact_count", "days_since_outreach", "gap_reason",
    ]].rename(columns={
        "merchant_id": "ID", "merchant_name": "Merchant", "industry": "Industry",
        "annual_payment_volume": "Volume", "contact_count": "Contacts",
        "days_since_outreach": "Days since outreach", "gap_reason": "Why flagged",
    })
    st.dataframe(
        show_rc, width="stretch", hide_index=True,
        column_config={
            "Volume": st.column_config.NumberColumn(format="$%.0f"),
            "Contacts": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    st.info(
        f"**So what:** **{money(at_risk_vol)}** of payment volume sits behind accounts that "
        f"are either going quiet or have lost their key contact. Re-engagement and contact "
        f"replacement protect that volume before a competitor or a routing change erodes it - "
        f"and they are the cheapest way to keep the Light Touch book covered.",
        icon=":material/lightbulb:",
    )


# ---------------------------------------------------------------------------
# View 5 - Methodology
# ---------------------------------------------------------------------------

def view_methodology(bundle: dict) -> None:
    s = bundle["summary"]
    pipe = bundle["pipeline"]
    st.subheader("Methodology & notes")

    st.markdown(
        f"""
**What this is.** A demonstration of how a Visa U.S. Merchant Business Development
analyst turns raw portfolio data into prioritisation and pipeline decisions for a
"Light Touch" book of ~300 merchant relationships. The data is **100% synthetic** -
generated on launch with a fixed seed so results are reproducible - and modeled on
the commercial levers a BD analyst works: Visa penetration, debit routing, co-brand,
value-added services (VAS), acceptance solutions, and data solutions.

**The dataset.** {s['n_merchants']} merchants across 8 industries and 5 regions, each
with annual payment volume, Visa penetration, relationship tier, engagement and
outreach history, contact coverage, and opportunity-eligibility flags. A linked
pipeline of commercial opportunities carries a type, funnel stage, estimated annual
value, and win probability. Each merchant is seeded with a behavioral archetype
(whitespace / well-covered / lapsed-contact / steady) so the prioritisation and
coverage logic can be validated against known ground truth (see `test_analytics.py`).

---

### How the priority score is built

Every merchant is scored 0-100 by a transparent, weighted combination of four
min-max-scaled signals:

| Component | Weight | Signal |
|---|---|---|
| **Volume** | 35% | Annual payment volume (bigger book = more at stake) |
| **Whitespace** | 30% | Volume x (1 - Visa penetration) - the un-penetrated dollars |
| **Opportunity** | 20% | Share of the 4 eligibility flags that are open |
| **Engagement gap** | 15% | Stale outreach + low engagement = being neglected |

High volume + LOW penetration + open flags + stale engagement pushes an account to
the top of the call list.

### How the pipeline is weighted

Each opportunity's weighted value = **estimated annual value x win probability**,
where probability comes from its funnel stage (Identified 10% -> Qualified 25% ->
Developing 45% -> Proposed 70%; Won = 100%, Lost = 0%). "Open" pipeline excludes the
terminal Won/Lost stages.

### How coverage gaps are flagged

A merchant is flagged when **either** its last outreach is older than 90 days **or**
its key contact is inactive (needs replacing). Flagged accounts are ranked by volume
so the biggest relationships at risk come first.

### What the current portfolio is telling us

- **{money(s['total_volume'])}** in annual payment volume, **{s['coverage_rate']:.0%}**
  of it on Visa today, leaving **{money(s['whitespace_volume'])}** of whitespace.
- **{s['n_open_opportunities']:,}** open opportunities worth **{money(pipe['total_weighted_open'])}**
  on a probability-weighted basis.
- A long tail of accounts has gone quiet or lost its key contact - cheap, high-return
  re-engagement work.

### Why it maps to the role

The "Analyst, U.S. Merchant Business Development" supports the Light Touch Merchant
Portfolio: prioritising accounts, identifying commercial opportunities (debit routing,
co-brand, VAS, acceptance), maintaining an opportunity pipeline, tracking engagement
and contact coverage, and turning portfolio data into prioritisation recommendations.
This dashboard produces exactly those deliverables.

---

*Honesty note: no real Visa or merchant data is used. This is a portfolio demonstration
of analytical approach, metric design, and merchant-portfolio judgment.*
"""
    )


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def main() -> None:
    st.sidebar.title("Merchant Portfolio & Opportunity Analytics")
    st.sidebar.caption("Visa U.S. Merchant BD - Light Touch portfolio")
    view = st.sidebar.radio(
        "View",
        [
            "Portfolio overview",
            "Account prioritization",
            "Opportunity pipeline",
            "Engagement & coverage",
            "Methodology",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Synthetic, seeded data generated on launch. A portfolio demonstration - "
        "no real Visa or merchant data is used."
    )

    if view == "Portfolio overview":
        view_overview(BUNDLE)
    elif view == "Account prioritization":
        view_prioritization(BUNDLE)
    elif view == "Opportunity pipeline":
        view_pipeline(BUNDLE)
    elif view == "Engagement & coverage":
        view_coverage(BUNDLE)
    else:
        view_methodology(BUNDLE)


if __name__ == "__main__":
    main()
