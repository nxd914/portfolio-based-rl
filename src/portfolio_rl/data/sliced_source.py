from __future__ import annotations

from dataclasses import dataclass

from portfolio_rl.data.base import DataSource
from portfolio_rl.data.types import BarData


@dataclass
class SlicedDataSource(DataSource):
    """DataSource wrapper exposing only a contiguous slice of data."""

    base: DataSource
    slc: slice

    def __post_init__(self) -> None:
        if self.slc.step not in (None, 1):
            raise ValueError("SlicedDataSource only supports step of 1")
        start = 0 if self.slc.start is None else self.slc.start
        stop = self.base.n_steps if self.slc.stop is None else self.slc.stop
        if start < 0 or stop < 0 or start > stop:
            raise ValueError("invalid slice bounds")
        if stop > self.base.n_steps:
            raise ValueError("slice stop exceeds base n_steps")
        self._start = int(start)
        self._stop = int(stop)

    @property
    def n_assets(self) -> int:
        return self.base.n_assets

    @property
    def n_steps(self) -> int:
        return self._stop - self._start

    def get_bar(self, t: int) -> BarData:
        if t < 0 or t >= self.n_steps:
            raise IndexError("t out of range for slice")
        return self.base.get_bar(self._start + t)
