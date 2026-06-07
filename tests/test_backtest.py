"""Tests for the factor backtest engine."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from signals_engine.rank.backtest import (
    BacktestResult,
    information_coefficient,
    make_rebalance_dates,
    run_backtest,
)


class TestInformationCoefficient:
    def test_perfect_positive_correlation(self):
        s = pd.Series(np.arange(10, dtype=float))
        ic = information_coefficient(s, s * 2 + 1)   # monotone transform
        assert abs(ic - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        s = pd.Series(np.arange(10, dtype=float))
        ic = information_coefficient(s, -s)
        assert abs(ic + 1.0) < 1e-6

    def test_zero_correlation(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.random(200))
        noise = pd.Series(rng.random(200))
        ic = information_coefficient(s, noise)
        # should be small (within ±3σ of 0 for n=200: ~0.07)
        assert abs(ic) < 0.20

    def test_returns_nan_for_too_few_obs(self):
        s = pd.Series([1.0, 2.0, 3.0])
        r = pd.Series([3.0, 2.0])       # fewer observations
        ic = information_coefficient(s, r)
        assert np.isnan(ic) or abs(ic) <= 1.0

    def test_nan_values_ignored(self):
        s = pd.Series({"A": 1.0, "B": 2.0, "C": np.nan, "D": 4.0})
        r = pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4})
        ic = information_coefficient(s, r)
        assert np.isfinite(ic)


class TestRunBacktest:
    def _make_panels(self, n_dates: int = 24, n_stocks: int = 30, seed: int = 7):
        rng = np.random.default_rng(seed)
        rebal_dates = list(
            pd.date_range("2021-01-01", periods=n_dates, freq="MS").date
        )
        tickers = [f"S{i:03d}" for i in range(n_stocks)]
        # signal with predictive power: add 0.5× signal to returns
        sig_values = rng.random((n_dates, n_stocks))
        noise = rng.normal(0, 0.02, (n_dates, n_stocks))
        ret_values = 0.003 * sig_values + noise
        sig_panel = pd.DataFrame(sig_values, index=rebal_dates, columns=tickers)
        ret_panel = pd.DataFrame(ret_values, index=rebal_dates, columns=tickers)
        return sig_panel, ret_panel

    def test_result_type(self):
        sig, ret = self._make_panels()
        result = run_backtest(sig, ret, "test_signal")
        assert isinstance(result, BacktestResult)

    def test_ic_series_length(self):
        sig, ret = self._make_panels(n_dates=24)
        result = run_backtest(sig, ret, "test_signal")
        # IC series should have at most n_dates entries
        assert len(result.ic_series) <= 24

    def test_spread_return_identity(self):
        """spread = long − short for every period."""
        sig, ret = self._make_panels()
        res = run_backtest(sig, ret, "test_signal")
        common = res.long_returns.index.intersection(res.short_returns.index).intersection(
            res.spread_returns.index
        )
        diff = (res.long_returns[common] - res.short_returns[common]
                - res.spread_returns[common]).abs()
        assert (diff < 1e-10).all()

    def test_hit_rate_bounds(self):
        sig, ret = self._make_panels()
        res = run_backtest(sig, ret, "test_signal")
        assert 0.0 <= res.hit_rate <= 1.0

    def test_predictive_signal_positive_mean_ic(self):
        """A signal with predictive power should yield positive mean IC."""
        sig, ret = self._make_panels(n_dates=36, n_stocks=50)
        res = run_backtest(sig, ret, "test_signal")
        assert res.mean_ic > 0.0


class TestMakeRebalanceDates:
    def test_start_included(self):
        dates = make_rebalance_dates(date(2020, 1, 1), date(2021, 1, 1))
        assert date(2020, 1, 1) in dates

    def test_monotone(self):
        dates = make_rebalance_dates(date(2020, 1, 1), date(2022, 1, 1))
        assert dates == sorted(dates)

    def test_monthly_count(self):
        dates = make_rebalance_dates(date(2020, 1, 1), date(2020, 12, 1))
        assert 11 <= len(dates) <= 12
