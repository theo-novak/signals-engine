"""Fundamental factor signals derived from XBRL financial statement data.

Each function accepts a tidy fundamentals DataFrame
(columns: ticker, period_end, revenue, net_income, eps_basic, total_assets,
total_debt, stockholders_equity, operating_cash_flow)
and returns a Series(ticker → signal_value) for the most recent available period
as of a given as-of date.

Convention: higher signal value = more desirable (e.g. earnings yield is positive,
accruals and leverage are sign-flipped so low raw value → high signal).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _latest_as_of(fund: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """For each ticker, keep the single most recent row with period_end <= as_of."""
    sub = fund[fund["period_end"] <= as_of].copy()
    idx = sub.groupby("ticker")["period_end"].idxmax()
    return sub.loc[idx].set_index("ticker")


def earnings_yield(
    fund: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.Series:
    """EPS / Price.  High = cheap relative to trailing earnings.

    prices: wide DataFrame (date × ticker) of adjusted closes.
    """
    latest = _latest_as_of(fund, as_of)
    price_row = prices[prices.index <= as_of].iloc[-1] if not prices.empty else pd.Series(dtype=float)
    eps = latest["eps_basic"].dropna()
    common = eps.index.intersection(price_row.index)
    p = price_row[common].replace(0, np.nan)
    ey = eps[common] / p
    return ey.rename("earnings_yield")


def asset_growth(fund: pd.DataFrame, as_of: date) -> pd.Series:
    """YoY growth in total assets.  High = aggressive balance-sheet expansion (negative signal)."""
    sub = fund[fund["period_end"] <= as_of].dropna(subset=["total_assets"]).copy()
    sub = sub.sort_values("period_end")
    last2 = sub.groupby("ticker").tail(2)

    result: dict[str, float] = {}
    for ticker, grp in last2.groupby("ticker"):
        if len(grp) < 2:
            continue
        a_prev, a_curr = grp["total_assets"].iloc[0], grp["total_assets"].iloc[1]
        if a_prev > 0:
            result[ticker] = (a_curr - a_prev) / a_prev
    return pd.Series(result, name="asset_growth")


def accruals(fund: pd.DataFrame, as_of: date) -> pd.Series:
    """Accruals = (Net income − Operating cash flow) / Total assets (sign-flipped).

    High accruals signal → lower earnings quality.
    We return the negative so that a high value means *good* quality.
    """
    latest = _latest_as_of(fund, as_of)
    ni = latest["net_income"]
    ocf = latest["operating_cash_flow"]
    ta = latest["total_assets"].replace(0, np.nan)
    raw = (ni - ocf) / ta
    return (-raw).rename("accruals")   # flip: low accrual → high quality signal


def return_on_assets(fund: pd.DataFrame, as_of: date) -> pd.Series:
    """Net income / Total assets.  Quality signal: higher = more profitable."""
    latest = _latest_as_of(fund, as_of)
    ta = latest["total_assets"].replace(0, np.nan)
    roa = latest["net_income"] / ta
    return roa.rename("return_on_assets")


def leverage(fund: pd.DataFrame, as_of: date) -> pd.Series:
    """Total debt / Total assets (sign-flipped so high signal = low leverage)."""
    latest = _latest_as_of(fund, as_of)
    ta = latest["total_assets"].replace(0, np.nan)
    raw = latest["total_debt"].fillna(0) / ta
    return (-raw).rename("leverage")


def sales_growth(fund: pd.DataFrame, as_of: date) -> pd.Series:
    """YoY revenue growth.  High = expanding top line."""
    sub = fund[fund["period_end"] <= as_of].dropna(subset=["revenue"]).copy()
    sub = sub.sort_values("period_end")
    last2 = sub.groupby("ticker").tail(2)

    result: dict[str, float] = {}
    for ticker, grp in last2.groupby("ticker"):
        if len(grp) < 2:
            continue
        r_prev, r_curr = grp["revenue"].iloc[0], grp["revenue"].iloc[1]
        if r_prev > 0:
            result[ticker] = (r_curr - r_prev) / r_prev
    return pd.Series(result, name="sales_growth")


def all_fundamental_signals(
    fund: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """Compute all fundamental signals as of a given date; return wide DataFrame."""
    signals = [
        earnings_yield(fund, prices, as_of),
        asset_growth(fund, as_of),
        accruals(fund, as_of),
        return_on_assets(fund, as_of),
        leverage(fund, as_of),
        sales_growth(fund, as_of),
    ]
    df = pd.concat(signals, axis=1)
    df.index.name = "ticker"
    return df
