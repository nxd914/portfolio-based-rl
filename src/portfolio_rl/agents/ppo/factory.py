from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from portfolio_rl.agents.ppo.policy import PPOPolicy
from portfolio_rl.agents.ppo.trainer import train_ppo


def make_ppo_factory(**ppo_kwargs) -> Callable[[int], Dict[str, Any]]:
    """Create a PPO agent factory compatible with walk-forward evaluation.

    Returns a callable that accepts a seed and returns a dict with keys:
    - agent: placeholder (training will return the trained agent)
    - train_fn: function(env, seed) -> trained PPOPolicy
    """

    def train_fn(env, seed: int):
        obs = env.reset(seed=seed)
        obs_vec = np.concatenate(
            [
                np.asarray(obs.features, dtype=float),
                np.asarray(obs.prices, dtype=float),
                np.asarray(obs.positions, dtype=float),
                np.asarray(obs.weights, dtype=float),
                np.array([float(obs.cash)], dtype=float),
            ],
            axis=0,
        )
        policy = PPOPolicy(obs_dim=obs_vec.shape[0], n_assets=obs.prices.shape[0], seed=seed)
        result = train_ppo(env, seed=seed, policy=policy, **ppo_kwargs)
        return result["policy"]

    def factory(seed: int):
        return {"agent": None, "train_fn": train_fn}

    return factory
