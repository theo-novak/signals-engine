"""Tests for cross-sectional ranking utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals_engine.rank.crosssection import (
    assign_quantiles,
    long_short_portfolio,
    rank_cross_section,
    rank_universe,
)


class TestRankCrossSection:
    def test_pct_rank_bounds(self):
        s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
        r = rank_cross_section(s)
        assert (r >= 0).all() and (r <= 1).all()

    def test_pct_rank_monotone(self):
        s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
        r = rank_cross_section(s)
        assert r["A"] < r["B"] < r["C"]

    def test_pct_rank_preserves_nan(self):
        s = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0})
        r = rank_cross_section(s)
        assert np.isnan(r["B"])
        assert np.isfinite(r["A"]) and np.isfinite(r["C"])

    def test_pct_rank_equal_values(self):
        s = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
        r = rank_cross_section(s)
        assert r.nunique() == 1      # all ties → same percentile


class TestAssignQuantiles:
    def test_quantile_labels_range(self):
        s = pd.Series(np.linspace(0, 1, 100))
        q = assign_quantiles(s, n=10)
        valid = q.dropna()
        assert valid.min() == 1
        assert valid.max() == 10

    def test_quantile_balanced(self):
        s = pd.Series(np.arange(100, dtype=float))
        q = assign_quantiles(s, n=5)
        counts = q.value_counts()
        # each decile should have ~20 stocks
        assert counts.std() < 5

    def test_quantile_preserves_nan(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        q = assign_quantiles(s, n=2)
        assert np.isnan(q.iloc[2])

    def test_quantile_too_few_unique(self):
        s = pd.Series([1.0] * 10)
        q = assign_quantiles(s, n=10)
        # fewer than n unique values → all NaN or a single bucket
        # implementation returns NaN for degenerate inputs
        assert q.isna().all() or q.nunique() == 1


class TestRankUniverse:
    def test_rank_universe_output_columns(self):
        df = pd.DataFrame({
            "signal_a": np.random.default_rng(0).random(10),
            "signal_b": np.random.default_rng(1).random(10),
        })
        out = rank_universe(df, n_quantiles=5)
        assert "signal_a_pct_rank" in out.columns
        assert "signal_b_pct_rank" in out.columns
        assert "signal_a_q5" in out.columns


class TestLongShortPortfolio:
    def test_spread_is_long_minus_short(self):
        rng = np.random.default_rng(99)
        signal = pd.Series(rng.random(50))
        returns = pd.Series(rng.normal(0, 0.02, 50))
        result = long_short_portfolio(signal, returns, n_quantiles=5)
        assert abs(result["spread_return"] - (result["long_return"] - result["short_return"])) < 1e-12

    def test_positive_signal_return_relationship(self):
        """Monotone signal with monotone returns → spread should be positive."""
        signal = pd.Series(np.linspace(0, 1, 50))
        returns = pd.Series(np.linspace(-0.01, 0.01, 50))    # higher signal → higher return
        result = long_short_portfolio(signal, returns, n_quantiles=5)
        assert result["spread_return"] > 0

    def test_insufficient_observations(self):
        signal = pd.Series([1.0, 2.0])
        returns = pd.Series([0.01, 0.02])
        result = long_short_portfolio(signal, returns, n_quantiles=5)
        assert np.isnan(result["spread_return"])
