from __future__ import annotations

import numpy as np


def normalize_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """Clip and scale weights to satisfy long-only and max weight constraints.

    Any leftover weight is implicitly held as cash by the environment.
    """

    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")

    clipped = np.clip(weights, 0.0, max_weight)
    return clipped
