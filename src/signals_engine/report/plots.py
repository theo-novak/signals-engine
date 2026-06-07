"""Plotly visualisations for the signals engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]


def plot_cumulative_spread(
    results: list,          # list[BacktestResult]
    title: str = "Cumulative Long-Short Spread Return",
) -> go.Figure:
    """Overlay cumulative spread returns for multiple signal backtests."""
    fig = go.Figure()
    for i, res in enumerate(results):
        cum = (1 + res.spread_returns.fillna(0)).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            name=res.signal_name,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.8),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Spread Return (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_ic_bar(
    result,     # BacktestResult
    title: str | None = None,
) -> go.Figure:
    """Monthly IC bar chart with ±ICIR envelope and rolling mean."""
    ic = result.ic_series.dropna()
    rolling = ic.rolling(6).mean()
    colors = ["#00CC96" if v >= 0 else "#EF553B" for v in ic.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ic.index, y=ic.values,
        name="Monthly IC",
        marker_color=colors,
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        name="6-month rolling IC",
        line=dict(color="#636EFA", width=2, dash="solid"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_layout(
        title=title or f"Information Coefficient — {result.signal_name}",
        xaxis_title="Date",
        yaxis_title="IC (Spearman)",
        template="plotly_white",
    )
    return fig


def plot_quantile_returns(
    signal: pd.Series,
    fwd_returns: pd.Series,
    n_quantiles: int = 5,
    title: str = "Mean Forward Return by Signal Quantile",
) -> go.Figure:
    """Bar chart of mean forward return for each signal quantile."""
    common = signal.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < n_quantiles * 2:
        return go.Figure()
    sig = signal[common]
    ret = fwd_returns[common]
    try:
        q = pd.qcut(sig, n_quantiles, labels=[f"Q{i+1}" for i in range(n_quantiles)],
                    duplicates="drop")
    except Exception:
        return go.Figure()
    means = ret.groupby(q).mean() * 100
    colors = [_PALETTE[2] if v >= 0 else _PALETTE[1] for v in means.values]
    fig = go.Figure(go.Bar(x=means.index.astype(str), y=means.values,
                           marker_color=colors))
    fig.update_layout(
        title=title,
        xaxis_title="Signal Quantile",
        yaxis_title="Mean Forward Return (%)",
        template="plotly_white",
    )
    return fig


def plot_ic_heatmap(
    ic_series: pd.Series,
    title: str = "IC Heatmap by Year and Month",
) -> go.Figure:
    """Calendar heatmap of monthly IC values."""
    df = pd.DataFrame({
        "ic": ic_series,
        "year": [d.year for d in ic_series.index],
        "month": [d.month for d in ic_series.index],
    })
    _MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = df.pivot(index="year", columns="month", values="ic")
    pivot.columns = [_MONTH_NAMES[m - 1] for m in pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="IC"),
    ))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def plot_factor_exposures(
    signals_df: pd.DataFrame,
    tickers: list[str] | None = None,
    title: str = "Factor Exposures by Stock",
) -> go.Figure:
    """Heatmap of z-scored signal values across stocks (rows = stocks, cols = signals)."""
    df = signals_df.copy()
    if tickers:
        df = df.loc[df.index.isin(tickers)]
    z = (df - df.mean()) / df.std().replace(0, 1)
    fig = go.Figure(go.Heatmap(
        z=z.values,
        x=z.columns.tolist(),
        y=z.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Z-score"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Signal",
        yaxis_title="Ticker",
        template="plotly_white",
    )
    return fig
