from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from portfolio_rl.data.base import DataSource
from .base import FeatureTransformer


@dataclass
class CausalReturns(FeatureTransformer):
    """Causal simple returns using only data up to time t.

    If insufficient history is available (t < window), returns zeros.
    """

    window: int = 1

    def fit(self, data: DataSource) -> None:
        return None

    def transform(self, t: int, data: DataSource) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if t < self.window:
            return np.zeros(data.n_assets, dtype=float)

        bar_t = data.get_bar(t)
        bar_prev = data.get_bar(t - self.window)
        price_t = bar_t.close
        price_prev = bar_prev.close

        returns = price_t / price_prev - 1.0
        return returns.astype(float)


@dataclass
class RollingVolatility(FeatureTransformer):
    """Rolling volatility of causal returns using only data up to time t.

    Uses population standard deviation (ddof=0) of simple returns over the
    last `window` periods. If insufficient history is available (t < window),
    returns zeros.
    """

    window: int = 10

    def fit(self, data: DataSource) -> None:
        return None

    def transform(self, t: int, data: DataSource) -> np.ndarray:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if t < self.window:
            return np.zeros(data.n_assets, dtype=float)

        returns = []
        for idx in range(t - self.window + 1, t + 1):
            bar_t = data.get_bar(idx)
            bar_prev = data.get_bar(idx - 1)
            ret = bar_t.close / bar_prev.close - 1.0
            returns.append(ret)

        returns_arr = np.stack(returns, axis=0)
        vol = np.std(returns_arr, axis=0, ddof=0)
        return vol.astype(float)
