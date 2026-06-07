"""Factor backtest engine.

Evaluates whether a signal explains cross-sectional returns over time.
Key metrics:
  - IC (Information Coefficient): Spearman rank correlation between signal and fwd returns
  - ICIR (IC Information Ratio): mean IC / std IC
  - Quantile spread returns: long top decile, short bottom decile
  - Turnover: fraction of portfolio that changes at each rebalance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class BacktestResult:
    signal_name: str
    rebalance_dates: list[date]
    ic_series: pd.Series                    # IC per rebalance date
    spread_returns: pd.Series               # long-short spread per period
    long_returns: pd.Series
    short_returns: pd.Series
    mean_ic: float
    icir: float
    hit_rate: float                         # fraction of periods with IC > 0
    cum_spread_return: float
    turnover_mean: float


def information_coefficient(
    signal: pd.Series,
    fwd_returns: pd.Series,
) -> float:
    """Spearman rank correlation between signal and forward returns."""
    common = signal.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < 5:
        return np.nan
    corr, _ = spearmanr(signal[common], fwd_returns[common])
    return float(corr)


def run_backtest(
    signals_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    signal_col: str,
    n_quantiles: int = 5,
) -> BacktestResult:
    """Time-series backtest of a single signal.

    signals_panel: (rebalance_date × ticker) with signal values in column signal_col.
                   Expects a MultiIndex (date, ticker) or wide (date × ticker) format.
                   If column signal_col is not present, expects signals_panel to already
                   be a wide date × ticker DataFrame of signal values.
    returns_panel: wide (date × ticker) forward return DataFrame.

    Both panels should share the same rebalance dates in their index.
    """
    # Normalize: expect signals_panel to be wide (date × ticker)
    if signal_col in signals_panel.columns:
        # Long format with date index and ticker columns implied by unstack
        sig_wide = signals_panel[signal_col].unstack("ticker") if isinstance(
            signals_panel.index, pd.MultiIndex
        ) else signals_panel[[signal_col]]
    else:
        sig_wide = signals_panel

    common_dates = sig_wide.index.intersection(returns_panel.index)
    if len(common_dates) < 3:
        raise ValueError(
            "signals_panel and returns_panel share fewer than 3 common dates"
        )

    ic_vals: dict[date, float] = {}
    long_rets: dict[date, float] = {}
    short_rets: dict[date, float] = {}
    spread_rets: dict[date, float] = {}
    prev_portfolio: set[str] = set()
    turnovers: list[float] = []

    for d in sorted(common_dates):
        sig_row = sig_wide.loc[d].dropna()
        ret_row = returns_panel.loc[d].dropna()
        common = sig_row.index.intersection(ret_row.index)
        if len(common) < n_quantiles * 2:
            continue

        sig = sig_row[common]
        ret = ret_row[common]

        # IC
        ic_vals[d] = information_coefficient(sig, ret)

        # Quantile spread
        try:
            q = pd.qcut(sig, n_quantiles, labels=False, duplicates="drop") + 1
        except Exception:
            continue

        long_mask = q == q.max()
        short_mask = q == q.min()
        long_rets[d] = float(ret[long_mask].mean())
        short_rets[d] = float(ret[short_mask].mean())
        spread_rets[d] = long_rets[d] - short_rets[d]

        # Turnover (fraction of long portfolio that changed)
        curr_long = set(sig[long_mask].index.tolist())
        if prev_portfolio:
            union = prev_portfolio | curr_long
            if union:
                turnovers.append(len(prev_portfolio.symmetric_difference(curr_long)) / len(union))
        prev_portfolio = curr_long

    ic_series = pd.Series(ic_vals, name="ic")
    spread_series = pd.Series(spread_rets, name="spread_return")

    mean_ic = float(ic_series.mean())
    icir = float(mean_ic / ic_series.std()) if ic_series.std() > 0 else np.nan
    hit_rate = float((ic_series > 0).mean())
    cum_spread = float((1 + spread_series.fillna(0)).prod() - 1)

    return BacktestResult(
        signal_name=signal_col,
        rebalance_dates=list(common_dates),
        ic_series=ic_series,
        spread_returns=spread_series,
        long_returns=pd.Series(long_rets, name="long_return"),
        short_returns=pd.Series(short_rets, name="short_return"),
        mean_ic=mean_ic,
        icir=icir,
        hit_rate=hit_rate,
        cum_spread_return=cum_spread,
        turnover_mean=float(np.mean(turnovers)) if turnovers else np.nan,
    )


def build_monthly_signal_panel(
    fund: pd.DataFrame,
    prices: pd.DataFrame,
    signal_fn,
    rebalance_dates: list[date],
) -> pd.DataFrame:
    """Call signal_fn(fund, prices, as_of) at each rebalance date to build a panel.

    Returns a wide DataFrame (date × ticker).
    """
    rows: dict[date, pd.Series] = {}
    for d in rebalance_dates:
        try:
            s = signal_fn(fund, prices, d)
            rows[d] = s
        except Exception:
            continue
    return pd.DataFrame(rows).T


def make_rebalance_dates(
    start: date,
    end: date,
    freq: str = "MS",
) -> list[date]:
    """Generate monthly (or other frequency) rebalance dates between start and end."""
    idx = pd.date_range(str(start), str(end), freq=freq)
    return [d.date() for d in idx]
