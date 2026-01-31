from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from portfolio_rl.config import ExecutionConfig
from portfolio_rl.data.types import BarData
from .base import ExecutionModel, ExecutionResult


@dataclass
class SimpleExecutionModel(ExecutionModel):
    """Deterministic execution with proportional commission and slippage.

    Fills are assumed immediate at the current bar price field with
    linear costs. No volume impact or partial fills are modeled.
    """

    config: ExecutionConfig = ExecutionConfig()

    def execute(self, trade_quantity: np.ndarray, bar: BarData) -> ExecutionResult:
        price = getattr(bar, self.config.price_field)
        trade_notional = trade_quantity * price

        commission = float(np.abs(trade_notional).sum()) * self.config.commission_rate
        slippage = float(np.abs(trade_notional).sum()) * self.config.slippage_rate
        total_cost = commission + slippage

        effective_price = price + np.sign(trade_quantity) * price * self.config.slippage_rate

        return ExecutionResult(
            filled_quantity=trade_quantity,
            total_cost=total_cost,
            commission=commission,
            slippage=slippage,
            effective_price=effective_price,
        )
