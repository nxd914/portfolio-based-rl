from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from portfolio_rl.env.types import Observation
from .policy import PPOPolicy
from .buffer import RolloutBuffer


def _normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.size == 0:
        return x
    mean = x.mean()
    std = x.std()
    if std < eps:
        return x - mean
    return (x - mean) / (std + eps)


def train_ppo(
    env,
    seed: int,
    num_iterations: int = 2,
    rollout_length: int = 8,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_range: float = 0.2,
    lr: float = 1e-2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.0,
    max_weight: float = 1.0,
    policy: Optional[PPOPolicy] = None,
) -> Dict[str, Any]:
    """Minimal PPO training scaffold.

    Returns dict with episode_returns, policy_loss, value_loss, entropy.
    """

    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    obs_vec = None

    if obs_vec is None:
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

    if policy is None:
        policy = PPOPolicy(obs_dim=obs_vec.shape[0], n_assets=obs.prices.shape[0], max_weight=max_weight, seed=seed)
    buffer = RolloutBuffer(gamma=gamma, lam=lam)

    episode_returns = []
    policy_losses = []
    value_losses = []
    entropies = []

    for _ in range(num_iterations):
        buffer.clear()
        ep_return = 0.0
        steps = 0
        done = False

        while steps < rollout_length and not done:
            obs_vec = policy.obs_to_vector(obs)
            action, logp, value = policy.sample(obs, rng)
            logits = policy.last_logits

            step = env.step(action)
            ep_return += step.reward

            buffer.add(
                obs_vec=obs_vec,
                action=logits,
                reward=step.reward,
                done=step.done,
                logp=logp,
                value=value,
                info=step.info,
            )

            obs = step.observation
            done = step.done
            steps += 1

        last_value = 0.0 if done else policy.value(obs)
        data = buffer.compute_gae(last_value=last_value)

        obs_batch = data["obs"]
        action_batch = data["actions"]
        old_logp = data["logps"]
        advantages = _normalize(data["advantages"])
        returns = data["returns"]

        new_logp, values, entropy = policy.evaluate(obs_batch, action_batch)

        ratio = np.exp(new_logp - old_logp)
        adv = advantages
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv

        policy_loss = -float(np.mean(np.minimum(surr1, surr2)))
        value_loss = float(np.mean((returns - values) ** 2))
        entropy_mean = float(np.mean(entropy))

        # Policy gradient w.r.t logp
        active = np.ones_like(adv, dtype=bool)
        active &= ~((adv >= 0) & (ratio > 1.0 + clip_range))
        active &= ~((adv < 0) & (ratio < 1.0 - clip_range))

        dlogp = np.zeros_like(adv)
        dlogp[active] = -(adv[active] * ratio[active]) / adv.shape[0]

        mu = obs_batch @ policy.W_mean.T + policy.b_mean
        std = np.exp(policy.log_std)
        inv_var = 1.0 / (std ** 2)

        diff = (action_batch - mu) * inv_var
        dmu = (dlogp[:, None]) * diff

        # log_std gradient
        dlog_std = dlogp[:, None] * (-1.0 + ((action_batch - mu) ** 2) * inv_var)
        dlog_std = dlog_std.mean(axis=0)

        # Value head gradient
        dvalue = (values - returns) / returns.shape[0]
        dW_value = (dvalue[:, None] * obs_batch).sum(axis=0, keepdims=True)
        db_value = np.array([dvalue.sum()])

        # Policy head gradient
        dW_mean = dmu.T @ obs_batch
        db_mean = dmu.sum(axis=0)

        # Entropy bonus (encourage exploration)
        dlog_std += -ent_coef

        # Apply gradients
        policy.W_mean -= lr * dW_mean
        policy.b_mean -= lr * db_mean
        policy.W_value -= lr * vf_coef * dW_value
        policy.b_value -= lr * vf_coef * db_value
        policy.log_std -= lr * dlog_std

        episode_returns.append(ep_return)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropies.append(entropy_mean)

        if done:
            obs = env.reset(seed=seed)

    return {
        "episode_returns": np.asarray(episode_returns, dtype=float),
        "policy_loss": np.asarray(policy_losses, dtype=float),
        "value_loss": np.asarray(value_losses, dtype=float),
        "entropy": np.asarray(entropies, dtype=float),
        "policy": policy,
    }
