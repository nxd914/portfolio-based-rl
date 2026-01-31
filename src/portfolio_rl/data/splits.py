from __future__ import annotations

from typing import List, Tuple


def walk_forward_splits(
    n_steps: int,
    train_size: int,
    test_size: int,
    step_size: int,
    embargo: int = 0,
) -> List[Tuple[slice, slice]]:
    """Generate walk-forward train/test splits with an optional embargo.

    Splits advance by step_size until no full test window remains.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size, test_size, step_size must be positive")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    splits: List[Tuple[slice, slice]] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_steps:
            break
        splits.append((slice(train_start, train_end), slice(test_start, test_end)))
        start += step_size
    return splits
