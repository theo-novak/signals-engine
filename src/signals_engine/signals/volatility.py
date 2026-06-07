"""Volatility-based signal filters.

These are not standalone alpha signals but *filters* applied on top of fundamental
or momentum signals: low-vol stocks have historically higher risk-adjusted returns
(the low-volatility anomaly, Ang et al. 2006).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def realized_vol(
    prices: pd.DataFrame,
    as_of: date,
    window: int = 63,
) -> pd.Series:
    """Trailing realized volatility (annualised daily std of log returns).

    window: trading days used (default 63 ≈ 3 months).
    """
    as_of_ts = pd.Timestamp(as_of)
    idx_series = pd.to_datetime(pd.Series(prices.index))
    mask = idx_series <= as_of_ts
    if mask.sum() < window + 1:
        return pd.Series(dtype=float, name="realized_vol")
    end_i = int(mask.values.nonzero()[0][-1])
    start_i = max(end_i - window, 0)
    window_prices = prices.iloc[start_i : end_i + 1]
    log_rets = np.log(window_prices / window_prices.shift(1)).dropna()
    rvol = log_rets.std() * np.sqrt(_TRADING_DAYS)
    return rvol.rename("realized_vol")


def idiosyncratic_vol(
    prices: pd.DataFrame,
    market_ticker: str,
    as_of: date,
    window: int = 63,
) -> pd.Series:
    """Idiosyncratic (residual) volatility relative to a market proxy.

    Regresses each stock's returns on the market returns and returns the
    annualised std of the residuals — a measure of stock-specific risk.
    Sign-flipped: high value → low idio vol → more desirable.
    """
    from sklearn.linear_model import LinearRegression

    as_of_ts = pd.Timestamp(as_of)
    idx_series = pd.to_datetime(pd.Series(prices.index))
    mask = idx_series <= as_of_ts
    if mask.sum() < window + 1 or market_ticker not in prices.columns:
        return pd.Series(dtype=float, name="idiosyncratic_vol")

    end_i = int(mask.values.nonzero()[0][-1])
    start_i = max(end_i - window, 0)
    rets = np.log(prices.iloc[start_i : end_i + 1] / prices.iloc[start_i : end_i + 1].shift(1)).dropna()

    mkt = rets[[market_ticker]].values
    result: dict[str, float] = {}
    for col in rets.columns:
        if col == market_ticker:
            continue
        y = rets[col].values.reshape(-1, 1)
        valid = np.isfinite(y.ravel()) & np.isfinite(mkt.ravel())
        if valid.sum() < 20:
            continue
        lr = LinearRegression().fit(mkt[valid], y[valid])
        resid = y[valid] - lr.predict(mkt[valid])
        result[col] = float(np.std(resid) * np.sqrt(_TRADING_DAYS))

    raw = pd.Series(result, name="idiosyncratic_vol")
    return (-raw).rename("idiosyncratic_vol")   # flip: low idio vol → high signal


def low_vol_filter(
    signal: pd.Series,
    rvol: pd.Series,
    vol_percentile_cap: float = 0.75,
) -> pd.Series:
    """Zero out or NaN-mask stocks in the top vol_percentile_cap of realized vol.

    Use this to apply a volatility screen on top of any factor signal.
    """
    common = signal.index.intersection(rvol.index)
    sig = signal[common].copy()
    vols = rvol[common]
    cap = vols.quantile(vol_percentile_cap)
    sig[vols > cap] = np.nan
    return sig
