import numpy as np
import pytest

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import Action, PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.eval import run_episode, cumulative_return, max_drawdown, realized_volatility, sharpe_ratio
from portfolio_rl.agents import RandomAgent


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


def test_rollout_deterministic_with_seed():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])

    env_a = make_env(data)
    env_b = make_env(data)

    agent_a = RandomAgent(seed=123)
    agent_b = RandomAgent(seed=123)

    out_a = run_episode(env_a, agent_a)
    out_b = run_episode(env_b, agent_b)

    assert np.allclose(out_a["rewards"], out_b["rewards"])
    assert np.allclose(out_a["transaction_costs"], out_b["transaction_costs"])
    assert np.allclose(out_a["turnover"], out_b["turnover"])
    assert np.allclose(out_a["cash"], out_b["cash"])


def test_metrics_handcomputed():
    values = np.array([1.0, 1.1, 1.0, 1.2])
    returns = np.array([0.1, -0.0909090909, 0.2])

    expected_cum = 0.2
    expected_dd = (1.1 - 1.0) / 1.1

    expected_vol = np.std(returns, ddof=0) * np.sqrt(252)
    expected_sharpe = (np.mean(returns) / np.std(returns, ddof=0)) * np.sqrt(252)

    assert cumulative_return(values) == pytest.approx(expected_cum)
    assert max_drawdown(values) == pytest.approx(expected_dd)
    assert realized_volatility(returns) == pytest.approx(expected_vol)
    assert sharpe_ratio(returns) == pytest.approx(expected_sharpe)


def test_run_episode_shapes():
    data = make_data([[10.0], [11.0], [12.0]])
    env = make_env(data)
    agent = RandomAgent(seed=0)

    out = run_episode(env, agent, max_steps=2)

    assert out["rewards"].ndim == 1
    assert out["portfolio_values"].ndim == 1
    assert out["transaction_costs"].ndim == 1
    assert out["turnover"].ndim == 1
    assert out["cash"].ndim == 1
    assert out["gross_exposure"].ndim == 1
    assert out["net_exposure"].ndim == 1
    assert out["weights_sum"].ndim == 1
    assert out["cash_frac"].ndim == 1

    n = out["rewards"].shape[0]
    assert n > 0
    assert out["portfolio_values"].shape[0] == n
    assert out["transaction_costs"].shape[0] == n
    assert out["turnover"].shape[0] == n
    assert out["cash"].shape[0] == n
    assert out["gross_exposure"].shape[0] == n
    assert out["net_exposure"].shape[0] == n
    assert out["weights_sum"].shape[0] == n
    assert out["cash_frac"].shape[0] == n
    assert isinstance(out["last_info"], dict)
    for key in ["gross_exposure", "net_exposure", "weights_sum", "cash_frac"]:
        assert np.isfinite(out[key]).all()
