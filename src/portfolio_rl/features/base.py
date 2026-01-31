from __future__ import annotations

from typing import Protocol
import numpy as np

from portfolio_rl.data.base import DataSource


class FeatureTransformer(Protocol):
    """Causal feature transformer interface."""

    def fit(self, data: DataSource) -> None:  # pragma: no cover - interface only
        ...

    def transform(self, t: int, data: DataSource) -> np.ndarray:  # pragma: no cover
        ...
