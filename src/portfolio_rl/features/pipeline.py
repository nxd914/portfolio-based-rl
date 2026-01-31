from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from portfolio_rl.data.base import DataSource
from .base import FeatureTransformer


@dataclass
class FeaturePipeline:
    """Compose multiple causal feature transformers."""

    transformers: Sequence[FeatureTransformer]

    def fit(self, data: DataSource) -> None:
        for transformer in self.transformers:
            transformer.fit(data)

    def transform(self, t: int, data: DataSource) -> np.ndarray:
        features = [transformer.transform(t, data) for transformer in self.transformers]
        if not features:
            return np.empty((0,), dtype=float)
        return np.concatenate(features, axis=0)
