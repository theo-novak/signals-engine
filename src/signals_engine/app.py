"""Streamlit dashboard for the cross-sectional signals engine.

Tabs:
  1. Universe & Data     — configure tickers, show fundamentals data quality
  2. Factor Explorer     — inspect signal values and cross-sectional rankings
  3. Backtester          — IC time series, quantile returns, cumulative spread
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signals_engine.data.store import init_schema, read_fundamentals, read_prices
from signals_engine.data.prices import compute_forward_returns
from signals_engine.signals.fundamentals import all_fundamental_signals
from signals_engine.signals.momentum import all_momentum_signals
from signals_engine.signals.volatility import realized_vol
from signals_engine.signals.composite import (
    FUNDAMENTALS_COMPOSITE, QUALITY_VALUE, MOMENTUM_QUALITY,
)
from signals_engine.rank.crosssection import rank_universe, long_short_portfolio
from signals_engine.rank.backtest import (
    make_rebalance_dates, build_monthly_signal_panel, run_backtest,
)
from signals_engine.report.plots import (
    plot_cumulative_spread, plot_ic_bar, plot_quantile_returns,
    plot_ic_heatmap, plot_factor_exposures,
)

st.set_page_config(page_title="Signals Engine", layout="wide")

_DEFAULT_UNIVERSE = "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,WFC,JNJ,UNH,XOM,CVX,PG,KO,PEP,HD,LOW,CAT,GE,MMM"
_DB_PATH = "data/signals.duckdb"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tickers_raw = st.text_area(
        "Universe (comma-separated)", _DEFAULT_UNIVERSE, height=120
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    as_of_date = st.date_input("Signal date (as-of)", value=date.today())
    horizon = st.selectbox("Forward return horizon (trading days)", [21, 63, 126, 252], index=0)
    n_quantiles = st.slider("Quantiles", 3, 10, 5)

    st.divider()
    st.subheader("Backtest range")
    bt_start = st.date_input("Start", value=date(date.today().year - 5, 1, 1))
    bt_end = st.date_input("End", value=date.today())

init_schema(_DB_PATH)

tab1, tab2, tab3 = st.tabs(
    ["📊 Universe & Data", "🔍 Factor Explorer", "📈 Backtester"]
)

# ── Tab 1: Universe & Data ────────────────────────────────────────────────────
with tab1:
    st.subheader("Fundamentals Coverage")
    fund = read_fundamentals(tickers, db_path=_DB_PATH)
    prices_wide = read_prices(tickers, db_path=_DB_PATH)

    if fund.empty:
        st.warning("No fundamentals data found. Run `signals fetch` from the CLI first.")
    else:
        coverage = (
            fund.groupby("ticker")
            .agg(
                periods=("period_end", "count"),
                earliest=("period_end", "min"),
                latest=("period_end", "max"),
                has_revenue=("revenue", lambda x: x.notna().sum()),
                has_eps=("eps_basic", lambda x: x.notna().sum()),
                has_ocf=("operating_cash_flow", lambda x: x.notna().sum()),
            )
            .reset_index()
        )
        st.dataframe(coverage, width="stretch")

    st.subheader("Price History")
    if prices_wide.empty:
        st.warning("No price data found. Run `signals fetch` from the CLI first.")
    else:
        st.write(f"{len(prices_wide)} trading days × {prices_wide.shape[1]} tickers "
                 f"({prices_wide.index[0]} → {prices_wide.index[-1]})")
        st.line_chart(
            prices_wide.div(prices_wide.iloc[0]).dropna(how="all"),
        )

# ── Tab 2: Factor Explorer ────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Signal Values — {as_of_date}")
    if fund.empty or prices_wide.empty:
        st.warning("Load data first (Tab 1 / CLI fetch).")
    else:
        fund_sigs = all_fundamental_signals(fund, prices_wide, as_of_date)
        mom_sigs = all_momentum_signals(prices_wide, as_of_date)
        rvol = realized_vol(prices_wide, as_of_date)
        all_sigs = pd.concat([fund_sigs, mom_sigs, rvol.rename("realized_vol")], axis=1)
        all_sigs = all_sigs.dropna(how="all")

        st.dataframe(
            all_sigs.style.background_gradient(cmap="RdYlGn", axis=0),
            width="stretch",
        )

        st.subheader("Factor Exposure Heatmap")
        if not all_sigs.empty:
            fig_exp = plot_factor_exposures(all_sigs.drop(columns=["realized_vol"], errors="ignore"))
            st.plotly_chart(fig_exp, width="stretch")

        st.subheader("Signal Rankings")
        rank_df = rank_universe(all_sigs.drop(columns=["realized_vol"], errors="ignore"),
                                n_quantiles=n_quantiles)
        st.dataframe(rank_df, width="stretch")

        st.subheader("Composite Signals")
        composites = {}
        for comp in [FUNDAMENTALS_COMPOSITE, QUALITY_VALUE, MOMENTUM_QUALITY]:
            try:
                composites[comp.name] = comp.build(all_sigs)
            except KeyError:
                pass
        if composites:
            comp_df = pd.DataFrame(composites).sort_values(
                list(composites.keys())[0], ascending=False
            )
            st.dataframe(comp_df, width="stretch")

# ── Tab 3: Backtester ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Signal Backtester")
    if fund.empty or prices_wide.empty:
        st.warning("Load data first (Tab 1 / CLI fetch).")
    else:
        _SIGNAL_OPTIONS = [
            "earnings_yield", "return_on_assets", "accruals",
            "leverage", "sales_growth", "asset_growth", "momentum_12_1",
        ]
        chosen_signals = st.multiselect(
            "Signals to backtest", _SIGNAL_OPTIONS,
            default=["earnings_yield", "return_on_assets"],
        )

        if st.button("Run Backtest"):
            from signals_engine.signals.fundamentals import (
                earnings_yield as ey, return_on_assets as roa,
                accruals as acc, leverage as lev,
                sales_growth as sg, asset_growth as ag,
            )
            from signals_engine.signals.momentum import momentum_12_1 as mom

            _FN_MAP = {
                "earnings_yield":   lambda f, p, d: ey(f, p, d),
                "return_on_assets": lambda f, p, d: roa(f, d),
                "accruals":         lambda f, p, d: acc(f, d),
                "leverage":         lambda f, p, d: lev(f, d),
                "sales_growth":     lambda f, p, d: sg(f, d),
                "asset_growth":     lambda f, p, d: ag(f, d),
                "momentum_12_1":    lambda f, p, d: mom(p, d),
            }

            fwd_ret = compute_forward_returns(prices_wide, horizon=horizon)
            rebal_dates = make_rebalance_dates(bt_start, bt_end)

            results = []
            for sig_name in chosen_signals:
                fn = _FN_MAP[sig_name]
                with st.spinner(f"Building {sig_name} panel..."):
                    panel = build_monthly_signal_panel(fund, prices_wide, fn, rebal_dates)
                    res = run_backtest(panel, fwd_ret, sig_name, n_quantiles=n_quantiles)
                    results.append(res)

            # Summary table
            rows = []
            for r in results:
                rows.append({
                    "Signal": r.signal_name,
                    "Mean IC": f"{r.mean_ic:.4f}",
                    "ICIR": f"{r.icir:.2f}" if pd.notna(r.icir) else "—",
                    "Hit Rate": f"{r.hit_rate:.1%}",
                    "Cum. Spread": f"{r.cum_spread_return:.1%}",
                    "Turnover": f"{r.turnover_mean:.1%}" if pd.notna(r.turnover_mean) else "—",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Signal"), width="stretch")

            # Cumulative spread chart
            if results:
                fig_spread = plot_cumulative_spread(results)
                st.plotly_chart(fig_spread, width="stretch")

            # Per-signal detail
            for r in results:
                with st.expander(f"Detail — {r.signal_name}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_ic = plot_ic_bar(r)
                        st.plotly_chart(fig_ic, width="stretch")
                    with c2:
                        if not r.ic_series.dropna().empty:
                            fig_hm = plot_ic_heatmap(r.ic_series.dropna())
                            st.plotly_chart(fig_hm, width="stretch")
