from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from portfolio_rl.env.types import Action, Observation


@dataclass
class HoldAgent:
    """Holds the previous action; starts with all zeros."""

    _last_action: Optional[Action] = None

    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        if self._last_action is None:
            weights = np.zeros(observation.prices.shape[0], dtype=float)
            self._last_action = Action(weights=weights)
        return self._last_action


@dataclass
class EqualWeightAgent:
    """Allocates equal weights up to max_weight per asset."""

    max_weight: float = 1.0

    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        n_assets = observation.prices.shape[0]
        weight = min(self.max_weight, 1.0 / n_assets)
        weights = np.full(n_assets, weight, dtype=float)
        return Action(weights=weights)


@dataclass
class CashOnlyAgent:
    """Always stays in cash (all zeros)."""

    def act(self, observation: Observation, deterministic: bool = False) -> Action:
        weights = np.zeros(observation.prices.shape[0], dtype=float)
        return Action(weights=weights)
