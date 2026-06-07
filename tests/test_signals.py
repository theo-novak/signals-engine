"""Tests for fundamental, momentum, and composite signal functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals_engine.signals.fundamentals import (
    accruals,
    all_fundamental_signals,
    asset_growth,
    earnings_yield,
    leverage,
    return_on_assets,
    sales_growth,
)
from signals_engine.signals.momentum import momentum_12_1, reversal_1m
from signals_engine.signals.composite import (
    FUNDAMENTALS_COMPOSITE,
    QUALITY_VALUE,
    CompositeSignal,
)


class TestFundamentalSignals:
    def test_earnings_yield_positive_for_profitable(self, fund_df, prices_df, as_of):
        ey = earnings_yield(fund_df, prices_df, as_of)
        assert not ey.empty
        # for profitable firms (positive EPS and positive price), EY should be positive
        assert (ey.dropna() > 0).all()

    def test_earnings_yield_index_is_tickers(self, fund_df, prices_df, as_of):
        ey = earnings_yield(fund_df, prices_df, as_of)
        assert ey.index.name == "ticker" or set(ey.index).issubset(set(fund_df["ticker"].unique()))

    def test_roa_non_negative_for_profitable_firms(self, fund_df, as_of):
        roa = return_on_assets(fund_df, as_of)
        assert not roa.empty
        # all synthetic firms have positive net income → ROA should be positive
        assert (roa.dropna() > 0).all()

    def test_asset_growth_bounded(self, fund_df, as_of):
        ag = asset_growth(fund_df, as_of)
        assert not ag.empty
        # synthetic firms have up to +15% asset growth, never more than 100%
        assert (ag.dropna().abs() < 1.0).all()

    def test_accruals_returns_same_tickers(self, fund_df, as_of):
        acc = accruals(fund_df, as_of)
        roa = return_on_assets(fund_df, as_of)
        assert set(acc.dropna().index) == set(roa.dropna().index)

    def test_leverage_is_negative_of_raw(self, fund_df, as_of):
        """leverage() is sign-flipped so high signal = low raw leverage."""
        latest = fund_df[fund_df["period_end"] <= as_of].sort_values("period_end")
        latest = latest.groupby("ticker").tail(1).set_index("ticker")
        raw_lev = latest["total_debt"] / latest["total_assets"]
        sig = leverage(fund_df, as_of)
        common = sig.dropna().index.intersection(raw_lev.index)
        pd.testing.assert_series_equal(
            sig[common].sort_index(),
            (-raw_lev[common]).sort_index(),
            check_names=False,
            rtol=1e-6,
        )

    def test_sales_growth_requires_two_periods(self, fund_df, as_of):
        # Filter to a single period per ticker — should produce empty
        single_period = fund_df[fund_df["period_end"] == fund_df["period_end"].max()]
        sg = sales_growth(single_period, as_of)
        assert sg.empty

    def test_all_fundamental_signals_shape(self, fund_df, prices_df, as_of):
        df = all_fundamental_signals(fund_df, prices_df, as_of)
        assert set(df.columns) >= {"earnings_yield", "return_on_assets", "accruals",
                                    "leverage", "asset_growth", "sales_growth"}
        assert len(df) <= len(fund_df["ticker"].unique())


class TestMomentumSignals:
    def test_momentum_returns_series(self, prices_df, as_of):
        mom = momentum_12_1(prices_df, as_of)
        assert isinstance(mom, pd.Series)
        assert not mom.empty

    def test_momentum_coverage(self, prices_df, as_of):
        """All tickers with sufficient history should have a momentum value."""
        mom = momentum_12_1(prices_df, as_of)
        # 3 years of data, 12-1 month lookback → all tickers should have values
        assert mom.dropna().shape[0] == prices_df.shape[1]

    def test_reversal_is_sign_flipped_vs_raw_1m(self, prices_df, as_of):
        """reversal_1m should be the negative of the raw 1-month return."""
        as_of_ts = pd.Timestamp(as_of)
        idx = pd.to_datetime(pd.Series(prices_df.index))
        end_i = int((idx <= as_of_ts).values.nonzero()[0][-1])
        p_end = prices_df.iloc[end_i]
        p_start = prices_df.iloc[max(end_i - 21, 0)]
        raw = p_end / p_start - 1
        rev = reversal_1m(prices_df, as_of)
        common = raw.dropna().index.intersection(rev.dropna().index)
        pd.testing.assert_series_equal(
            rev[common].sort_index(),
            (-raw[common]).sort_index(),
            check_names=False,
            rtol=1e-6,
        )

    def test_reversal_insufficient_history(self, prices_df):
        """Only 10 days of data → no reversal signal."""
        short = prices_df.iloc[:10]
        rev = reversal_1m(short, short.index[-1])
        assert rev.empty


class TestCompositeSignal:
    def test_composite_build_returns_series(self, fund_df, prices_df, as_of):
        df = all_fundamental_signals(fund_df, prices_df, as_of)
        score = FUNDAMENTALS_COMPOSITE.build(df)
        assert isinstance(score, pd.Series)
        assert not score.empty

    def test_composite_respects_weights(self, fund_df, prices_df, as_of):
        """A single-signal composite should rank identically to the raw signal."""
        df = all_fundamental_signals(fund_df, prices_df, as_of)
        single = CompositeSignal(name="test", weights={"return_on_assets": 1.0})
        score = single.build(df)
        roa = df["return_on_assets"].dropna()
        common = score.dropna().index.intersection(roa.index)
        # Rank order must be identical
        assert score[common].rank().corr(roa[common].rank()) > 0.99

    def test_composite_missing_signal_raises(self, fund_df, prices_df, as_of):
        df = all_fundamental_signals(fund_df, prices_df, as_of)
        bad = CompositeSignal(name="bad", weights={"nonexistent_signal": 1.0})
        with pytest.raises(KeyError):
            bad.build(df)

    def test_quality_value_composite(self, fund_df, prices_df, as_of):
        df = all_fundamental_signals(fund_df, prices_df, as_of)
        score = QUALITY_VALUE.build(df)
        assert score.dropna().shape[0] >= 5
