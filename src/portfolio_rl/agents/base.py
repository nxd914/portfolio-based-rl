from __future__ import annotations

from typing import Protocol

from portfolio_rl.env.types import Action, Observation


class Agent(Protocol):
    """Agent interface for policy evaluation and learning."""

    def act(self, observation: Observation, deterministic: bool = False) -> Action:  # pragma: no cover
        ...
