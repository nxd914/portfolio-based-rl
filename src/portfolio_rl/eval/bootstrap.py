from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    x: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI over the first dimension.

    Returns (nan, nan) for empty input.
    """

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(stat_fn(x[idx]))

    lower = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper
