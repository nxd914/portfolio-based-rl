from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .types import BarData


@dataclass
class ArrayDataSource:
    """In-memory data source backed by NumPy arrays.

    Arrays must be shaped (n_steps, n_assets).
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamps: Optional[Sequence[pd.Timestamp]] = None

    def __post_init__(self) -> None:
        self._validate()

    @property
    def n_assets(self) -> int:
        return int(self.close.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.close.shape[0])

    def get_bar(self, t: int) -> BarData:
        if t < 0 or t >= self.n_steps:
            raise IndexError("t out of range")
        timestamp = None
        if self.timestamps is not None:
            timestamp = self.timestamps[t]
        return BarData(
            open=self.open[t],
            high=self.high[t],
            low=self.low[t],
            close=self.close[t],
            volume=self.volume[t],
            timestamp=timestamp,
        )

    def _validate(self) -> None:
        shapes = {arr.shape for arr in [self.open, self.high, self.low, self.close, self.volume]}
        if len(shapes) != 1:
            raise ValueError("OHLCV arrays must share the same shape")
        if self.close.ndim != 2:
            raise ValueError("OHLCV arrays must be 2D (n_steps, n_assets)")
        if self.timestamps is not None and len(self.timestamps) != self.close.shape[0]:
            raise ValueError("timestamps length must match n_steps")
