from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_rl.data import ArrayDataSource, walk_forward_splits
from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline
from portfolio_rl.features.basic import CausalReturns, RollingVolatility
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.baselines import CashOnlyAgent, EqualWeightAgent
from portfolio_rl.agents.ppo.factory import make_ppo_factory
from portfolio_rl.experiments.walk_forward import run_walk_forward
from portfolio_rl.experiments.reporting import make_run_id


def load_data(path: Path):
    data = np.load(path)
    return ArrayDataSource(
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
    )


def make_env(data):
    features = FeaturePipeline([
        CausalReturns(window=1),
        RollingVolatility(window=10),
    ])
    execution_model = SimpleExecutionModel(config=ExecutionConfig())
    reward_fn = LogReturnReward(cost_weight=1.0)
    config = EnvConfig(initial_cash=1_000_000.0)
    return PortfolioEnv(
        data=data,
        features=features,
        execution_model=execution_model,
        reward_fn=reward_fn,
        config=config,
    )


def main():
    data_path = Path("data/real/ohlcv.npz")
    if not data_path.exists():
        raise FileNotFoundError("data/real/ohlcv.npz not found. Run scripts/fetch_stooq.py")

    data = load_data(data_path)

    train_size = 504
    test_size = 252
    step_size = 252
    embargo = 5

    splits = walk_forward_splits(
        n_steps=data.n_steps,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        embargo=embargo,
    )

    agents = {
        "cash": lambda seed: CashOnlyAgent(),
        "equal": lambda seed: EqualWeightAgent(),
        "ppo": make_ppo_factory(num_iterations=5, rollout_length=64),
    }

    out_base = Path("reports/real")
    out_base.mkdir(parents=True, exist_ok=True)

    for name, factory in agents.items():
        run_id = make_run_id(prefix=name, seed=0, config={
            "train_size": train_size,
            "test_size": test_size,
            "step_size": step_size,
            "embargo": embargo,
            "n_steps": data.n_steps,
        })
        out_dir = out_base / name / run_id
        results = run_walk_forward(
            data_source=data,
            make_env_fn=make_env,
            agent_factory=factory,
            seed=0,
            splits=splits,
            max_steps=test_size,
            out_dir=str(out_dir),
            run_metadata={
                "agent": name,
                "seed": 0,
                "train_size": train_size,
                "test_size": test_size,
                "step_size": step_size,
                "embargo": embargo,
                "n_steps": data.n_steps,
            },
        )
        summary = results["summary"]["metrics_mean"]
        fields = results["summary"]["fields_mean"]
        print(
            f"{name}: cumret={summary.get('cumulative_return'):.3f} "
            f"sharpe={summary.get('sharpe_ratio'):.2f} "
            f"turn={fields.get('mean_turnover'):.4f}"
        )


if __name__ == "__main__":
    main()
