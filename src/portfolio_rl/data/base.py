from __future__ import annotations

from typing import Protocol

from .types import BarData


class DataSource(Protocol):
    """Abstract data source. Must not expose future data."""

    @property
    def n_assets(self) -> int:  # pragma: no cover - interface only
        ...

    @property
    def n_steps(self) -> int:  # pragma: no cover - interface only
        ...

    def get_bar(self, t: int) -> BarData:  # pragma: no cover - interface only
        ...
