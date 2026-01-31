from __future__ import annotations

from typing import Protocol

from portfolio_rl.env.types import Action, Observation


class Expert(Protocol):
    """Expert suggestion interface (does not execute trades)."""

    def suggest(self, observation: Observation) -> Action:  # pragma: no cover
        ...
