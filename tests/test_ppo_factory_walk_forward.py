import numpy as np

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.ppo.factory import make_ppo_factory
from portfolio_rl.experiments.walk_forward import run_walk_forward
from portfolio_rl.data.splits import walk_forward_splits


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


def test_make_ppo_factory_deterministic():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0], [13.0, 15.0]])
    env_a = make_env(data)
    env_b = make_env(data)

    factory = make_ppo_factory(num_iterations=1, rollout_length=3)

    obj_a = factory(0)
    obj_b = factory(0)

    policy_a = obj_a["train_fn"](env_a, seed=0)
    policy_b = obj_b["train_fn"](env_b, seed=0)

    assert np.allclose(policy_a.W_mean, policy_b.W_mean)
    assert np.allclose(policy_a.b_mean, policy_b.b_mean)
    assert np.allclose(policy_a.W_value, policy_b.W_value)
    assert np.allclose(policy_a.b_value, policy_b.b_value)
    assert np.allclose(policy_a.log_std, policy_b.log_std)


def test_walk_forward_with_ppo_runs():
    data = make_data([10.0, 10.2, 10.1, 10.3, 10.4, 10.5, 10.6, 10.7])
    splits = walk_forward_splits(n_steps=data.n_steps, train_size=4, test_size=2, step_size=4, embargo=0)

    out = run_walk_forward(
        data_source=data,
        make_env_fn=make_env,
        agent_factory=make_ppo_factory(num_iterations=1, rollout_length=2),
        seed=0,
        splits=splits,
        max_steps=2,
    )

    assert "splits" in out
    assert "summary" in out
    assert len(out["splits"]) == 1

    split_res = out["splits"][0]["result"]
    assert "metrics" in split_res
    for key in ["cumulative_return", "max_drawdown", "realized_volatility", "sharpe_ratio"]:
        assert np.isfinite(split_res["metrics"][key])
