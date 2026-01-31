import numpy as np
import pytest

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.ppo.policy import PPOPolicy
from portfolio_rl.agents.ppo.buffer import RolloutBuffer
from portfolio_rl.agents.ppo.trainer import train_ppo


class CloseFeature(FeatureTransformer):
    def fit(self, data):
        return None

    def transform(self, t, data):
        return data.get_bar(t).close.astype(float)


def make_data(close_values):
    close = np.array(close_values, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)
    return ArrayDataSource(open=open_, high=high, low=low, close=close, volume=volume)


def make_env(data):
    features = FeaturePipeline([CloseFeature()])
    execution_model = SimpleExecutionModel(config=ExecutionConfig())
    reward_fn = LogReturnReward(cost_weight=1.0)
    config = EnvConfig(initial_cash=1000.0)
    return PortfolioEnv(
        data=data,
        features=features,
        execution_model=execution_model,
        reward_fn=reward_fn,
        config=config,
    )


def test_ppo_deterministic():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])

    env_a = make_env(data)
    env_b = make_env(data)

    out_a = train_ppo(env_a, seed=123, num_iterations=2, rollout_length=4)
    out_b = train_ppo(env_b, seed=123, num_iterations=2, rollout_length=4)

    assert np.allclose(out_a["episode_returns"], out_b["episode_returns"])


def test_ppo_action_valid():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])
    env = make_env(data)
    obs = env.reset()

    obs_vec = np.concatenate(
        [
            obs.features,
            obs.prices,
            obs.positions,
            obs.weights,
            np.array([obs.cash], dtype=float),
        ]
    )

    policy = PPOPolicy(obs_dim=obs_vec.shape[0], n_assets=obs.prices.shape[0], seed=0)
    action = policy.act(obs)

    assert action.weights.shape == (obs.prices.shape[0],)
    assert np.isfinite(action.weights).all()
    assert np.all(action.weights >= 0.0)
    rng = np.random.default_rng(0)
    action_s, logp, value = policy.sample(obs, rng)
    assert np.isfinite(logp)
    assert np.isfinite(value)
    assert action_s.weights.shape == (obs.prices.shape[0],)
    assert np.isfinite(action_s.weights).all()
    assert np.all(action_s.weights >= 0.0)


def test_gae_shapes():
    buffer = RolloutBuffer(gamma=0.99, lam=0.95)
    obs = np.array([1.0, 2.0])
    action = np.array([0.1, 0.2])

    for i in range(3):
        buffer.add(obs, action, reward=1.0, done=False, logp=0.0, value=0.5)

    data = buffer.compute_gae(last_value=0.5)

    assert data["advantages"].shape == (3,)
    assert data["returns"].shape == (3,)
    assert np.isfinite(data["advantages"]).all()
    assert np.isfinite(data["returns"]).all()


def test_policy_act_deterministic():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])
    env = make_env(data)
    obs = env.reset()

    obs_vec = np.concatenate(
        [
            obs.features,
            obs.prices,
            obs.positions,
            obs.weights,
            np.array([obs.cash], dtype=float),
        ]
    )

    policy = PPOPolicy(obs_dim=obs_vec.shape[0], n_assets=obs.prices.shape[0], seed=0)
    action_a = policy.act(obs, deterministic=True)
    action_b = policy.act(obs, deterministic=True)

    assert np.allclose(action_a.weights, action_b.weights)
