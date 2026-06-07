"""Cross-sectional ranking utilities.

Stocks are ranked within a universe based on a signal value. Rankings can be expressed
as percentile ranks (0–1) or quantile bucket labels (e.g. deciles 1–10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_cross_section(signal: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank within the cross-section (0 = worst, 1 = best).

    NaN values are excluded and remain NaN in the output.
    ascending=True: higher signal value → higher rank (default for most signals
    where high is good, e.g. earnings yield, ROA).
    """
    return signal.rank(pct=True, na_option="keep", ascending=ascending).rename(
        f"{signal.name}_pct_rank"
    )


def assign_quantiles(
    signal: pd.Series,
    n: int = 10,
    labels: bool = True,
) -> pd.Series:
    """Assign quantile buckets 1 (bottom) to n (top).

    Uses pd.qcut on valid observations; NaN inputs stay NaN.
    """
    valid = signal.dropna()
    if valid.empty or valid.nunique() < n:
        return pd.Series(np.nan, index=signal.index, name=f"{signal.name}_q{n}")
    bucket = pd.qcut(valid, n, labels=False, duplicates="drop") + 1
    return bucket.reindex(signal.index).rename(f"{signal.name}_q{n}")


def rank_universe(
    signals_df: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Rank all signal columns in signals_df cross-sectionally.

    Returns a DataFrame with both pct_rank and quantile columns for each signal.
    """
    out: dict[str, pd.Series] = {}
    for col in signals_df.columns:
        out[f"{col}_pct_rank"] = rank_cross_section(signals_df[col])
        out[f"{col}_q{n_quantiles}"] = assign_quantiles(signals_df[col], n=n_quantiles)
    return pd.DataFrame(out)


def long_short_portfolio(
    signal: pd.Series,
    fwd_returns: pd.Series,
    n_quantiles: int = 5,
) -> dict[str, float]:
    """Split universe into quantiles and return the top-minus-bottom spread return.

    Returns a dict with 'long_return', 'short_return', 'spread_return'.
    """
    common = signal.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < n_quantiles * 2:
        return {"long_return": np.nan, "short_return": np.nan, "spread_return": np.nan}

    sig = signal[common]
    ret = fwd_returns[common]
    q = pd.qcut(sig, n_quantiles, labels=False, duplicates="drop") + 1

    long_ret = float(ret[q == q.max()].mean())
    short_ret = float(ret[q == q.min()].mean())
    return {
        "long_return": long_ret,
        "short_return": short_ret,
        "spread_return": long_ret - short_ret,
    }
