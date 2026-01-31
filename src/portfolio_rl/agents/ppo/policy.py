from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from portfolio_rl.env.types import Action, Observation
from portfolio_rl.agents.base import Agent


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _softplus_inverse(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.maximum(y, eps)
    return np.log(np.expm1(y))


def _gaussian_logp(a: np.ndarray, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    var = np.exp(2.0 * log_std)
    return -0.5 * (((a - mu) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi))


def _gaussian_entropy(log_std: np.ndarray) -> np.ndarray:
    return log_std + 0.5 * (1.0 + np.log(2.0 * np.pi))


@dataclass
class PPOPolicy(Agent):
    """Minimal linear Gaussian policy with value head.

    The policy samples unconstrained logits from a diagonal Gaussian and
    transforms them into non-negative weights via softplus, then clips
    by max_weight. It does not enforce sum-to-one.
    """

    obs_dim: int
    n_assets: int
    max_weight: float = 1.0
    init_std: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W_mean = rng.normal(scale=0.01, size=(self.n_assets, self.obs_dim))
        self.b_mean = np.zeros(self.n_assets, dtype=float)
        self.W_value = rng.normal(scale=0.01, size=(1, self.obs_dim))
        self.b_value = np.zeros(1, dtype=float)
        self.log_std = np.full(self.n_assets, np.log(self.init_std), dtype=float)
        self._last_logits = None

    def obs_to_vector(self, obs: Observation) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs.features, dtype=float),
                np.asarray(obs.prices, dtype=float),
                np.asarray(obs.positions, dtype=float),
                np.asarray(obs.weights, dtype=float),
                np.array([float(obs.cash)], dtype=float),
            ],
            axis=0,
        )

    def _forward(self, obs_vec: np.ndarray) -> Tuple[np.ndarray, float]:
        mu = self.W_mean @ obs_vec + self.b_mean
        value = float((self.W_value @ obs_vec + self.b_value).item())
        return mu, value

    def act(self, obs: Observation, deterministic: bool = False) -> Action:
        obs_vec = self.obs_to_vector(obs)
        mu, value = self._forward(obs_vec)
        std = np.exp(self.log_std)
        eps = np.zeros(self.n_assets, dtype=float)
        logits = mu + std * eps
        weights = _softplus(logits)
        weights = np.clip(weights, 0.0, self.max_weight)
        return Action(weights=weights.astype(float))

    def sample(
        self,
        obs: Observation,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> Tuple[Action, float, float]:
        obs_vec = self.obs_to_vector(obs)
        mu, value = self._forward(obs_vec)
        std = np.exp(self.log_std)
        eps = np.zeros(self.n_assets, dtype=float) if deterministic else rng.normal(size=self.n_assets)
        logits = mu + std * eps
        logp = float(_gaussian_logp(logits, mu, self.log_std).sum())
        weights = _softplus(logits)
        weights = np.clip(weights, 0.0, self.max_weight)
        self._last_logits = logits.copy()
        return Action(weights=weights.astype(float)), logp, value

    def evaluate(self, obs_batch: np.ndarray, action_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate logp, value, entropy for a batch.

        action_batch is expected to be the sampled logits (pre-softplus).
        """

        mu = obs_batch @ self.W_mean.T + self.b_mean
        values = (obs_batch @ self.W_value.T + self.b_value).reshape(-1)
        logp = _gaussian_logp(action_batch, mu, self.log_std).sum(axis=1)
        entropy = _gaussian_entropy(self.log_std).sum() * np.ones_like(logp)
        return logp, values, entropy

    def value(self, obs: Observation) -> float:
        obs_vec = self.obs_to_vector(obs)
        return float((self.W_value @ obs_vec + self.b_value).item())

    @property
    def last_logits(self) -> np.ndarray:
        if self._last_logits is None:
            raise RuntimeError("No action has been sampled yet.")
        return self._last_logits

    def weights_to_logits(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, 0.0, self.max_weight)
        return _softplus_inverse(weights)
