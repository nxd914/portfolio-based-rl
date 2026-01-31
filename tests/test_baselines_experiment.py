import numpy as np
import pytest

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.baselines import HoldAgent, EqualWeightAgent, CashOnlyAgent
from portfolio_rl.experiments.run import run_experiment
from portfolio_rl.eval.rollout import run_episode


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


def test_baselines_actions_valid():
    data = make_data([[10.0, 12.0], [11.0, 13.0]])
    env = make_env(data)
    obs = env.reset()

    agents = [HoldAgent(), EqualWeightAgent(), CashOnlyAgent()]
    for agent in agents:
        action = agent.act(obs)
        assert action.weights.shape == (2,)
        assert np.isfinite(action.weights).all()
        assert np.all(action.weights >= 0.0)


def test_run_experiment_keys_and_metrics():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])
    env = make_env(data)
    agent = EqualWeightAgent()

    out = run_experiment(env, agent, seed=0, max_steps=2)

    assert "metrics" in out
    for key in ["cumulative_return", "max_drawdown", "realized_volatility", "sharpe_ratio"]:
        assert key in out["metrics"]
        assert np.isfinite(out["metrics"][key])

    for key in [
        "mean_transaction_cost",
        "mean_turnover",
        "mean_gross_exposure",
        "mean_net_exposure",
        "mean_weights_sum",
        "final_cash",
        "final_portfolio_value",
    ]:
        assert key in out
        assert np.isfinite(out[key])

    rollout = run_episode(make_env(data), agent, max_steps=2)
    assert out["mean_transaction_cost"] == pytest.approx(np.mean(rollout["transaction_costs"]))
    assert out["mean_turnover"] == pytest.approx(np.mean(rollout["turnover"]))
    assert out["mean_gross_exposure"] == pytest.approx(np.mean(rollout["gross_exposure"]))
    assert out["mean_net_exposure"] == pytest.approx(np.mean(rollout["net_exposure"]))
    assert out["mean_weights_sum"] == pytest.approx(np.mean(rollout["weights_sum"]))


def test_cash_only_zero_turnover_and_costs():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])
    env = make_env(data)
    agent = CashOnlyAgent()

    out = run_experiment(env, agent, seed=0, max_steps=2)
    assert out["mean_turnover"] == pytest.approx(0.0)
    assert out["mean_transaction_cost"] == pytest.approx(0.0)
