from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class Action:
    """Portfolio action expressed as target weights per asset."""

    weights: np.ndarray


@dataclass(frozen=True)
class Observation:
    """Environment observation returned to the agent."""

    features: np.ndarray
    prices: np.ndarray
    positions: np.ndarray
    cash: float
    weights: np.ndarray


@dataclass(frozen=True)
class StepResult:
    """Result of a single environment step."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
