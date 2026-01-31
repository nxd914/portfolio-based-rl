import numpy as np
import pytest

from portfolio_rl.data import ArrayDataSource
from portfolio_rl.data.splits import walk_forward_splits
from portfolio_rl.data.sliced_source import SlicedDataSource
from portfolio_rl.experiments.walk_forward import run_walk_forward
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


def test_walk_forward_splits_nonoverlap():
    splits = walk_forward_splits(n_steps=20, train_size=5, test_size=3, step_size=4, embargo=2)
    for train_slc, test_slc in splits:
        assert train_slc.stop <= test_slc.start
        assert test_slc.start == train_slc.stop + 2
        assert 0 <= train_slc.start < train_slc.stop <= 20
        assert 0 <= test_slc.start < test_slc.stop <= 20


def test_sliced_data_source_bounds():
    data = make_data([1.0, 2.0, 3.0, 4.0, 5.0])
    sliced = SlicedDataSource(base=data, slc=slice(1, 4))

    _ = sliced.get_bar(0)
    _ = sliced.get_bar(2)

    with pytest.raises(IndexError):
        _ = sliced.get_bar(-1)

    with pytest.raises(IndexError):
        _ = sliced.get_bar(3)


def test_walk_forward_runs_baseline():
    data = make_data([10.0, 10.5, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2])
    splits = walk_forward_splits(n_steps=8, train_size=3, test_size=2, step_size=3, embargo=1)

    out = run_walk_forward(
        data_source=data,
        make_env_fn=make_env,
        agent_factory=lambda seed: CashOnlyAgent(),
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

    summary = out["summary"]
    for key in ["metrics_mean", "metrics_std", "fields_mean", "fields_std"]:
        assert key in summary
