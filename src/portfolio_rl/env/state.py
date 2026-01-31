from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PortfolioState:
    """Portfolio holdings at a single time step."""

    cash: float
    positions: np.ndarray

    def value(self, prices: np.ndarray) -> float:
        return float(self.cash + np.dot(self.positions, prices))

    def weights(self, prices: np.ndarray) -> np.ndarray:
        total_value = self.value(prices)
        if total_value <= 0.0:
            return np.zeros_like(self.positions)
        return (self.positions * prices) / total_value
