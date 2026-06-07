"""Composite signal construction: combine multiple factor signals into one score.

Signals are first cross-sectionally z-scored (winsorized at 3σ), then linearly combined
with user-supplied weights. Missing values are handled per-signal before combining.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CompositeSignal:
    """A named combination of factor signals with weights."""

    name: str
    weights: dict[str, float]       # {signal_col: weight}
    winsor_sigma: float = 3.0       # winsorise at ±N std before z-scoring

    def build(self, signals_df: pd.DataFrame) -> pd.Series:
        """Return cross-sectional composite score for each ticker.

        signals_df: DataFrame(ticker × signal_name).
        """
        missing = [c for c in self.weights if c not in signals_df.columns]
        if missing:
            raise KeyError(f"Signals not found in DataFrame: {missing}")

        zscores: list[pd.Series] = []
        for col, w in self.weights.items():
            s = signals_df[col].dropna()
            if s.empty or s.std() == 0:
                continue
            # winsorise
            lo = s.mean() - self.winsor_sigma * s.std()
            hi = s.mean() + self.winsor_sigma * s.std()
            s = s.clip(lo, hi)
            # z-score
            z = (s - s.mean()) / s.std()
            zscores.append(z * w)

        if not zscores:
            return pd.Series(dtype=float, name=self.name)

        combined = pd.concat(zscores, axis=1).sum(axis=1, min_count=1)
        return combined.rename(self.name)


# Pre-built composite signal definitions

FUNDAMENTALS_COMPOSITE = CompositeSignal(
    name="fundamentals_composite",
    weights={
        "earnings_yield":    1.0,
        "return_on_assets":  1.0,
        "accruals":          0.5,
        "leverage":          0.5,
    },
)

QUALITY_VALUE = CompositeSignal(
    name="quality_value",
    weights={
        "earnings_yield":    1.0,
        "return_on_assets":  1.0,
        "accruals":          1.0,
        "leverage":          0.5,
        "asset_growth":     -0.5,   # penalise aggressive expanders
    },
)

MOMENTUM_QUALITY = CompositeSignal(
    name="momentum_quality",
    weights={
        "momentum_12_1":     1.0,
        "return_on_assets":  0.5,
        "accruals":          0.5,
    },
)
