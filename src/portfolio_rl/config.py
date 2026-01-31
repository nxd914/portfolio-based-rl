from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """Action interpretation for the environment."""

    TARGET_WEIGHTS = "target_weights"


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution and cost parameters."""

    commission_rate: float = 0.0005
    slippage_rate: float = 0.0005
    price_field: str = "close"


@dataclass(frozen=True)
class ConstraintConfig:
    """Portfolio constraints enforced by the environment."""

    long_only: bool = True
    max_weight: float = 1.0


@dataclass(frozen=True)
class EnvConfig:
    """Top-level environment configuration."""

    action_kind: ActionKind = ActionKind.TARGET_WEIGHTS
    execution: ExecutionConfig = ExecutionConfig()
    constraints: ConstraintConfig = ConstraintConfig()
    start_t: int = 0
    initial_cash: float = 1_000_000.0
