from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from portfolio_rl.data.types import BarData


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing a trade at a single step."""

    filled_quantity: np.ndarray
    total_cost: float
    commission: float
    slippage: float
    effective_price: np.ndarray


class ExecutionModel(Protocol):
    """Execution model interface. Must be deterministic given inputs."""

    def execute(self, trade_quantity: np.ndarray, bar: BarData) -> ExecutionResult:  # pragma: no cover
        ...
