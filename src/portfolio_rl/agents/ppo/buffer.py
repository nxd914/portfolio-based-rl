from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class RolloutBuffer:
    """Simple rollout buffer for PPO."""

    gamma: float = 0.99
    lam: float = 0.95

    def __post_init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.logps: List[float] = []
        self.values: List[float] = []
        self.infos: List[Dict[str, Any]] = []

    def add(
        self,
        obs_vec: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        logp: float,
        value: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.obs.append(obs_vec.astype(float))
        self.actions.append(action.astype(float))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logps.append(float(logp))
        self.values.append(float(value))
        self.infos.append(info or {})

    def compute_gae(self, last_value: float) -> Dict[str, np.ndarray]:
        rewards = np.asarray(self.rewards, dtype=float)
        dones = np.asarray(self.dones, dtype=float)
        values = np.asarray(self.values, dtype=float)
        n = rewards.shape[0]

        advantages = np.zeros(n, dtype=float)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values

        return {
            "obs": np.asarray(self.obs, dtype=float),
            "actions": np.asarray(self.actions, dtype=float),
            "logps": np.asarray(self.logps, dtype=float),
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logps.clear()
        self.values.clear()
        self.infos.clear()
