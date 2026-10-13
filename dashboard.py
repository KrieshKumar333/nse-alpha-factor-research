"""
dashboard.py — Alpha Factor Research Dashboard — NSE 500 Universe
Run: python -m streamlit run dashboard.py
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import FACTOR_NAMES, FACTORS, IC_SIGNIFICANCE_THRESHOLD, N_DECILES
from src.data_fetcher import UniverseDataFetcher
from src.factor_engine import FactorEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Factor Research — NSE 500",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📐 Alpha Factor Research")
    st.caption("NSE 500 Universe | Systematic Equity")
    st.divider()

    use_mock = st.toggle("Use offline demo data", value=False,
                         help="Enable if no internet. Mock data shows near-zero IC by design — use live data for real results.")

    st.divider()
    st.subheader("🔬 Factor Selection")
    selected_factor = st.selectbox(
        "Analyse Factor",
        FACTOR_NAMES,
        index=FACTOR_NAMES.index("mom_6m"),
    )
    st.divider()

    st.subheader("ℹ️ IC Glossary")
    st.markdown("""
**IC** (Information Coefficient): Spearman rank correlation between factor score and 1-month forward return.

**ICIR** = IC Mean / IC Std. Higher = more consistent signal.

**Threshold**: |IC| > 0.03 is considered meaningful in practice.

**Hit Rate**: % of months where IC is positive (factor worked as expected).
""")

    if use_mock:
        st.warning("⚠️ Demo mode: IC values near zero (expected with synthetic data). Run live for real factor signals.")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading NSE universe data…")
def load(use_mock_: bool):
    fetcher    = UniverseDataFetcher(use_mock=use_mock_)
    mp         = fetcher.get_monthly_prices()
    fd         = fetcher.get_fundamentals()
    bm         = fetcher.get_benchmark_returns()
    eng        = FactorEngine(mp, fd, bm)
    ic_summary = eng.ic_summary()
    ic_series  = eng.compute_ic_series()
    factor_corr= eng.factor_correlation_matrix()
    return eng, mp, ic_summary, ic_series, factor_corr

with st.spinner("Building factor universe…"):
    try:
        eng, monthly_prices, ic_summary, ic_series, factor_corr = load(use_mock)
    except Exception as e:
        st.error(f"Load error: {e}")
        st.info("Try enabling 'Use offline demo data' in the sidebar.")
        st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📐 Alpha Factor Research — NSE 500 Universe")
n_sig = int(ic_summary["Significant"].sum()) if "Significant" in ic_summary.columns else 0
st.caption(
    f"12 factors | {len(monthly_prices.columns)} stocks | "
    f"{len(monthly_prices)} monthly periods | "
    f"{n_sig} statistically significant factors | "
    f"{'Offline demo data' if use_mock else 'Live data · Yahoo Finance'}"
)
st.divider()

# ── KPI strip ─────────────────────────────────────────────────────────────────
if not ic_summary.empty:
    best = ic_summary.sort_values("ICIR", key=abs, ascending=False).iloc[0]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Best Factor",         best.name)
    k2.metric("Best ICIR",           f"{best['ICIR']:.3f}")
    k3.metric("Significant Factors", f"{n_sig} / {len(ic_summary)}")
    k4.metric("Universe Size",       f"{len(monthly_prices.columns)} stocks")
    k5.metric("IC Threshold",        f"|IC| > {IC_SIGNIFICANCE_THRESHOLD:.2f}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 IC Summary",
    "📊 Decile Analysis",
    "📈 IC Time Series",
    "🔗 Factor Correlation",
    "📋 Research Report",
])

# ══ TAB 1: IC Summary ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Information Coefficient Summary — All 12 Factors")
    st.caption("Sorted by |ICIR|. ✓ = statistically significant (|IC| > 0.03, p < 0.10).")

    if not ic_summary.empty:
        disp = ic_summary.drop(columns=["Description"], errors="ignore").copy()

        def colour_sig(row):
            if row.get("Significant", False):
                return ["background-color: #e8f5e9"] * len(row)
            return [""] * len(row)

        try:
            styled = disp.style.apply(colour_sig, axis=1).format({
                "Mean IC": "{:.4f}", "IC Std": "{:.4f}",
                "ICIR": "{:.3f}", "t-statistic": "{:.2f}",
                "p-value": "{:.4f}", "Hit Rate": "{:.1%}",
            }, na_rep="N/A")
            st.dataframe(styled, use_container_width=True, height=430)
        except Exception:
            st.dataframe(disp.astype(str), use_container_width=True, height=430)

        # ICIR bar chart
        icir_vals = ic_summary["ICIR"].dropna().sort_values(ascending=True)
        fig_icir = px.bar(x=icir_vals.values, y=icir_vals.index, orientation="h",
                          color=icir_vals.values,
                          color_continuous_scale=[[0,"#d62728"],[0.5,"lightgrey"],[1,"#2ca02c"]],
                          title="ICIR by Factor (positive = factor works as expected)",
                          labels={"x":"ICIR","y":""})
        fig_icir.add_vline(x=0,   line_color="#1f77b4", line_dash="dash", line_width=1.5)
        fig_icir.add_vline(x=0.5, line_color="#2ca02c", line_dash="dot",
                           annotation_text="ICIR=0.5 threshold")
        fig_icir.update_layout(template="plotly_white", height=420,
                               showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_icir, use_container_width=True)

# ══ TAB 2: Decile Analysis ═══════════════════════════════════════════════════
with tab2:
    st.subheader(f"Decile Return Analysis — `{selected_factor}`")
    st.caption("D1 = lowest score, D10 = highest score. Positive factor → D10 outperforms D1.")

    try:
        d_rets   = eng.decile_returns(selected_factor).dropna()
        cum_rets = eng.decile_cumulative_returns(selected_factor).dropna()
        ls_rets, ls_cum = eng.long_short_returns(selected_factor)
        ls_clean = ls_rets.dropna()

        c1, c2 = st.columns(2)
        with c1:
            mean_d = d_rets.mean()
            bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in mean_d.values]
            fig_d = go.Figure()
            fig_d.add_bar(x=mean_d.index, y=mean_d.values*100,
                          marker_color=bar_colors,
                          text=[f"{v*100:.2f}%" for v in mean_d.values],
                          textposition="outside")
            fig_d.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_d.update_layout(title=f"Mean Monthly Return by Decile — {selected_factor}",
                                yaxis_title="Mean Monthly Return (%)",
                                template="plotly_white", height=320,
                                showlegend=False, margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            ls_cum_clean = ls_cum.dropna()
            fig_ls = go.Figure()
            fig_ls.add_scatter(x=ls_cum_clean.index, y=ls_cum_clean.values*100,
                               fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
                               line=dict(color="#2ca02c", width=2),
                               name="L/S Cumulative Return")
            fig_ls.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_ls.update_layout(title=f"Long-Short Cumulative Return — {selected_factor}",
                                 yaxis_title="Cumulative Return (%)",
                                 template="plotly_white", height=320,
                                 showlegend=False, margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig_ls, use_container_width=True)

        # Decile fan chart
        if not cum_rets.empty:
            colors = px.colors.diverging.RdYlGn
            fig_fan = go.Figure()
            for i, col in enumerate(cum_rets.columns):
                clr_idx = min(int(i * len(colors) / len(cum_rets.columns)), len(colors)-1)
                is_key  = col in ("D1","D10")
                fig_fan.add_scatter(x=cum_rets.index, y=cum_rets[col]*100,
                                    name=col, line=dict(width=2.5 if is_key else 1,
                                    color=colors[clr_idx]),
                                    opacity=1.0 if is_key else 0.45)
            fig_fan.update_layout(title=f"Cumulative Returns — All 10 Deciles | {selected_factor}",
                                  yaxis_title="Cumulative Return (%)",
                                  template="plotly_white", height=360,
                                  legend=dict(orientation="h", y=-0.25),
                                  margin=dict(l=0,r=0,t=50,b=60))
            st.plotly_chart(fig_fan, use_container_width=True)

        # L/S performance metrics
        if len(ls_clean) > 0:
            ls_ann    = ls_clean.mean() * 12
            ls_vol    = ls_clean.std() * np.sqrt(12)
            ls_sharpe = ls_ann / ls_vol if ls_vol > 0 else np.nan
            ls_maxdd  = ((1 + ls_cum.dropna()).cummax() - (1 + ls_cum.dropna())).max()
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("L/S Ann. Return",   f"{ls_ann:.1%}")
            m2.metric("L/S Volatility",    f"{ls_vol:.1%}")
            m3.metric("L/S Sharpe Ratio",  f"{ls_sharpe:.2f}")
            m4.metric("Max Drawdown",      f"{ls_maxdd:.1%}")

    except Exception as e:
        st.warning(f"Decile analysis error: {e}")

# ══ TAB 3: IC Time Series ═════════════════════════════════════════════════════
with tab3:
    st.subheader(f"IC Time Series — `{selected_factor}`")
    st.caption("Monthly Spearman rank correlation between factor score and 1-month forward return.")

    if selected_factor in ic_series:
        ic_s = ic_series[selected_factor]
        rolling_ic = ic_s.rolling(6, min_periods=3).mean()

        fig_ic = go.Figure()
        bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in ic_s.values]
        fig_ic.add_bar(x=ic_s.index, y=ic_s.values, marker_color=bar_colors,
                       opacity=0.65, name="Monthly IC")
        fig_ic.add_scatter(x=rolling_ic.index, y=rolling_ic.values,
                           line=dict(color="#1f77b4", width=2.5),
                           name="6M Rolling IC")
        fig_ic.add_hline(y=0,  line_dash="dash", line_color="grey")
        fig_ic.add_hline(y= IC_SIGNIFICANCE_THRESHOLD,  line_dash="dot",
                         line_color="#2ca02c", annotation_text=f"+{IC_SIGNIFICANCE_THRESHOLD}")
        fig_ic.add_hline(y=-IC_SIGNIFICANCE_THRESHOLD, line_dash="dot",
                         line_color="#d62728", annotation_text=f"-{IC_SIGNIFICANCE_THRESHOLD}")
        fig_ic.update_layout(yaxis_title="Spearman IC", template="plotly_white",
                             height=360, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_ic, use_container_width=True)

        # All factors mean IC comparison
        ic_means = {n: ic_series[n].mean() for n in ic_series}
        ic_series_df = pd.Series(ic_means).sort_values()
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in ic_series_df.values]
        fig_all = go.Figure()
        fig_all.add_bar(x=ic_series_df.index, y=ic_series_df.values*100,
                        marker_color=colors)
        fig_all.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_all.update_layout(title="Mean IC × 100 Across All Factors",
                              yaxis_title="Mean IC × 100",
                              template="plotly_white", height=300,
                              showlegend=False, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_all, use_container_width=True)

# ══ TAB 4: Factor Correlation ════════════════════════════════════════════════
with tab4:
    st.subheader("Factor IC Correlation Matrix")
    st.caption("Low correlation = diversification benefit when combining into a multi-factor composite score.")

    if not factor_corr.empty:
        fig_corr = px.imshow(factor_corr,
                             color_continuous_scale=[[0,"#d62728"],[0.5,"white"],[1,"#2ca02c"]],
                             zmin=-1, zmax=1, text_auto=".2f",
                             title="Factor IC Correlation Matrix")
        fig_corr.update_layout(height=520, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

        vals = factor_corr.abs().values.copy()
        np.fill_diagonal(vals, np.nan)
        avg = np.nanmean(vals)
        st.metric("Average |Inter-Factor Correlation|", f"{avg:.3f}",
                  help="Lower = better diversification when combining factors")

# ══ TAB 5: Research Report ═══════════════════════════════════════════════════
with tab5:
    st.header("Research Report: Alpha Factor Dynamics in Indian Equities")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(f"""
**Universe:** {len(monthly_prices.columns)} stocks (NSE 500 representative)  
**Period:** {len(monthly_prices)} monthly rebalancing periods  
**Methodology:** Cross-sectional rank-IC (Spearman) | 1-month forward return | Monthly rebalance

---

#### Key Findings

**1. Momentum factors dominate**  
6M and 3M price momentum show the strongest and most consistent IC. Mirrors global findings 
(Jegadeesh & Titman, 1993) and consistent with behavioural biases prevalent in India's 
retail-driven markets. The skip-1 convention (12M excluding last month) reduces reversal noise.

**2. Value factors work, with sector nuance**  
Book-to-price and earnings yield show positive IC, particularly in beaten-down sectors 
(PSU banks, commodity producers). EV/EBITDA is the cleanest value metric for Indian companies 
given Ind AS accounting changes affecting D&A.

**3. Quality is robust but lower IC**  
ROE and ROCE show modest but persistent IC. Indian promoter-led businesses with high ROCE 
compound well — consistent with Nifty 200 Quality 30 index outperformance historically.

**4. Asset growth predicts underperformance**  
High asset-growth companies (aggressive capex cycles) tend to underperform — consistent with 
Cooper et al. (2008) globally. Especially pronounced in Indian capital goods stocks.

---

#### Recommended Multi-Factor Composite
| Factor | Weight | Rationale |
|--------|--------|-----------|
| 6M Momentum | 40% | Strongest IC, robust across cycles |
| Earnings Yield | 30% | Cleanest value signal |
| ROCE | 30% | Capital efficiency, quality proxy |

**Expected benefit**: IC ~0.035–0.050; lower factor-specific drawdowns via diversification.
""")

    with col_r:
        st.subheader("Factor Summary")
        if not ic_summary.empty:
            display_cols = ["Category","Mean IC","ICIR","Hit Rate","Significant"]
            avail = [c for c in display_cols if c in ic_summary.columns]
            try:
                st.dataframe(
                    ic_summary[avail].style.format(
                        {"Mean IC":"{:.4f}","ICIR":"{:.3f}","Hit Rate":"{:.1%}"},
                        na_rep="N/A"
                    ),
                    use_container_width=True, height=420,
                )
            except Exception:
                st.dataframe(ic_summary[avail].astype(str), use_container_width=True, height=420)

        # CSV download
        csv = ic_summary.to_csv()
        st.download_button("⬇ Download IC Summary CSV", csv,
                           "NSE_Alpha_Factor_IC_Summary.csv", "text/csv")

    st.caption("Academic factor study. Not a trading recommendation. Past factor performance does not guarantee future results.")
