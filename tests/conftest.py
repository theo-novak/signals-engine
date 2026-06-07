"""Shared fixtures for the signals engine test suite.

All fixtures are fully offline — no network or database access required.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(42)
_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "XOM", "PG", "JNJ", "HD"]
_N_DAYS = 756      # ~3 years of trading days
_N_TICKERS = len(_TICKERS)
_START = date(2022, 1, 3)


@pytest.fixture(scope="session")
def trade_dates() -> list[date]:
    return list(pd.bdate_range(str(_START), periods=_N_DAYS).date)


@pytest.fixture(scope="session")
def prices_df(trade_dates) -> pd.DataFrame:
    """Synthetic adjusted close price matrix — geometric random walk, seed 42."""
    log_returns = RNG.normal(0.0003, 0.015, size=(_N_DAYS, _N_TICKERS))
    cum_log = np.cumsum(log_returns, axis=0)
    start_prices = RNG.uniform(50, 500, size=_N_TICKERS)
    prices = start_prices * np.exp(cum_log)
    return pd.DataFrame(prices, index=trade_dates, columns=_TICKERS)


@pytest.fixture(scope="session")
def fund_df() -> pd.DataFrame:
    """Synthetic annual fundamentals — 5 fiscal years, all 10 tickers."""
    rows = []
    for ticker in _TICKERS:
        revenue_base = RNG.uniform(1e9, 5e10)
        asset_base = RNG.uniform(5e9, 2e11)
        for year in range(2019, 2024):
            rev = revenue_base * (1 + RNG.uniform(-0.05, 0.20))
            ni = rev * RNG.uniform(0.05, 0.25)
            ta = asset_base * (1 + RNG.uniform(-0.03, 0.15))
            rows.append({
                "ticker": ticker,
                "cik": f"{RNG.integers(1000, 9999):010d}",
                "period_end": date(year, 12, 31),
                "revenue": rev,
                "net_income": ni,
                "eps_basic": ni / RNG.integers(5e8, 2e9),
                "total_assets": ta,
                "total_debt": ta * RNG.uniform(0.1, 0.5),
                "stockholders_equity": ta * RNG.uniform(0.2, 0.6),
                "operating_cash_flow": ni * RNG.uniform(0.8, 1.4),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def as_of() -> date:
    return date(2023, 12, 29)
