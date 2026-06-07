"""Price-based momentum and reversal signals."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def momentum_12_1(
    prices: pd.DataFrame,
    as_of: date,
    skip_months: int = 1,
    formation_months: int = 12,
) -> pd.Series:
    """12-1 month momentum (Jegadeesh & Titman 1993).

    Return = cumulative return from t-13 to t-2 months.
    The most recent month is skipped to avoid the 1-month reversal contaminating the signal.
    """
    idx = prices.index
    as_of_ts = pd.Timestamp(as_of)
    # find the row at or just before as_of
    mask = pd.to_datetime(pd.Series(idx)) <= as_of_ts
    if mask.sum() == 0:
        return pd.Series(dtype=float, name="momentum_12_1")
    end_idx = int(mask.values.nonzero()[0][-1])

    # skip 1 month back (~21 trading days), then go back 12 months (~252 days total)
    skip_td = skip_months * 21
    form_td = formation_months * 21

    near_end = max(end_idx - skip_td, 0)
    near_start = max(near_end - form_td, 0)

    if near_end <= near_start:
        return pd.Series(dtype=float, name="momentum_12_1")

    p_start = prices.iloc[near_start]
    p_end = prices.iloc[near_end]
    ret = p_end / p_start.replace(0, np.nan) - 1.0
    return ret.rename("momentum_12_1")


def reversal_1m(prices: pd.DataFrame, as_of: date) -> pd.Series:
    """1-month short-term reversal (sign-flipped: high = recent loser, expected to bounce)."""
    idx = prices.index
    as_of_ts = pd.Timestamp(as_of)
    mask = pd.to_datetime(pd.Series(idx)) <= as_of_ts
    if mask.sum() < 22:
        return pd.Series(dtype=float, name="reversal_1m")
    end_idx = int(mask.values.nonzero()[0][-1])
    start_idx = max(end_idx - 21, 0)
    p_start = prices.iloc[start_idx]
    p_end = prices.iloc[end_idx]
    raw = p_end / p_start.replace(0, np.nan) - 1.0
    return (-raw).rename("reversal_1m")


def all_momentum_signals(prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
    signals = [
        momentum_12_1(prices, as_of),
        reversal_1m(prices, as_of),
    ]
    df = pd.concat(signals, axis=1)
    df.index.name = "ticker"
    return df
