import numpy as np
import pytest

from portfolio_rl.eval.metrics import sortino_ratio, calmar_ratio
from portfolio_rl.eval.bootstrap import bootstrap_ci
from portfolio_rl.experiments.walk_forward import run_walk_forward
from portfolio_rl.data import ArrayDataSource, walk_forward_splits
from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.baselines import CashOnlyAgent


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


def test_sortino_known():
    returns = np.array([0.1, -0.1, 0.0])
    expected = (np.mean(returns) / 0.1) * np.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_calmar_known():
    values = np.array([100.0, 110.0, 100.0, 120.0])
    dd = (110.0 - 100.0) / 110.0
    ann = (120.0 / 100.0) ** (252 / 3) - 1.0
    expected = ann / dd
    assert calmar_ratio(values) == pytest.approx(expected)


def test_bootstrap_ci_deterministic():
    x = np.arange(10.0)
    ci1 = bootstrap_ci(x, np.mean, n_boot=200, seed=0)
    ci2 = bootstrap_ci(x, np.mean, n_boot=200, seed=0)
    assert ci1 == ci2


def test_walk_forward_ci_keys_present():
    data = make_data([10.0, 10.2, 10.1, 10.3, 10.4, 10.5, 10.6, 10.7])
    splits = walk_forward_splits(n_steps=data.n_steps, train_size=4, test_size=2, step_size=2, embargo=0)
    out = run_walk_forward(
        data_source=data,
        make_env_fn=make_env,
        agent_factory=lambda seed: CashOnlyAgent(),
        seed=0,
        splits=splits,
        max_steps=2,
    )

    summary = out["summary"]["metrics_mean"]
    assert "mean_sharpe" in summary
    assert "ci_sharpe" in summary
    assert "mean_cumulative_return" in summary
    assert "ci_cumulative_return" in summary

    ci_sharpe = summary["ci_sharpe"]
    ci_cum = summary["ci_cumulative_return"]
    assert np.isfinite(ci_sharpe[0]) and np.isfinite(ci_sharpe[1])
    assert np.isfinite(ci_cum[0]) and np.isfinite(ci_cum[1])
