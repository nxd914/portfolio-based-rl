from __future__ import annotations

import random
import numpy as np


def seed_all(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs and return a Generator.

    This function is deterministic and does not touch any other global state.
    """

    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
