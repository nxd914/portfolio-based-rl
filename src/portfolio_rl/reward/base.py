from __future__ import annotations

from typing import Protocol


class RewardFn(Protocol):
    """Reward function interface."""

    def __call__(self, prev_value: float, next_value: float, transaction_cost: float) -> float:  # pragma: no cover
        ...
