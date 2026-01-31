import json
import numpy as np

from portfolio_rl.experiments.reporting import save_json
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


def test_save_json_handles_numpy(tmp_path):
    obj = {
        "scalar": np.float64(1.23),
        "array": np.array([1.0, 2.0]),
        "nested": {"int": np.int64(2)},
    }
    path = tmp_path / "out.json"
    save_json(path, obj)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["scalar"] == 1.23
    assert data["array"] == [1.0, 2.0]
    assert data["nested"]["int"] == 2


def test_run_walk_forward_writes_files(tmp_path):
    data = make_data([10.0, 10.2, 10.1, 10.3, 10.4, 10.5])
    splits = walk_forward_splits(n_steps=data.n_steps, train_size=3, test_size=2, step_size=3, embargo=0)

    out_dir = tmp_path / "wf"
    _ = run_walk_forward(
        data_source=data,
        make_env_fn=make_env,
        agent_factory=lambda seed: CashOnlyAgent(),
        seed=0,
        splits=splits,
        max_steps=2,
        out_dir=str(out_dir),
        run_metadata={"tag": "test"},
    )

    summary_path = out_dir / "summary.json"
    splits_path = out_dir / "splits.csv"
    config_path = out_dir / "config.json"

    assert summary_path.exists()
    assert splits_path.exists()
    assert config_path.exists()

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "summary" in summary
    assert "metadata" in summary

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["seed"] == 0
    assert "split_params" in config

    with splits_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "cumulative_return" in header
    assert "mean_transaction_cost" in header
