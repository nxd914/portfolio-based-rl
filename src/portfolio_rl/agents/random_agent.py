from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from portfolio_rl.env.types import Action, Observation


@dataclass
class RandomAgent:
    """Random long-only agent producing target weights.

    Weights are non-negative and finite, but are not rescaled to sum <= 1.
    Deterministic given the provided seed.
    """

    seed: Optional[int] = None
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        n_assets = observation.prices.shape[0]
        weights = self._rng.random(n_assets) * self.max_weight
        return Action(weights=weights.astype(float))
