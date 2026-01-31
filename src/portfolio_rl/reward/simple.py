from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import RewardFn


@dataclass(frozen=True)
class LogReturnReward(RewardFn):
    """Log return reward with explicit transaction cost penalty.

    Note: transaction costs already impact next_value; the cost_weight term
    makes that penalty explicit and adjustable for research purposes.
    """

    cost_weight: float = 1.0
    min_value: float = 1e-12

    def __call__(self, prev_value: float, next_value: float, transaction_cost: float) -> float:
        safe_prev = max(prev_value, self.min_value)
        safe_next = max(next_value, self.min_value)
        log_return = float(np.log(safe_next / safe_prev))
        cost_penalty = self.cost_weight * (transaction_cost / safe_prev)
        return log_return - cost_penalty
