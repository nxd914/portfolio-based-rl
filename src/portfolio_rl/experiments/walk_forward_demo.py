from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource, walk_forward_splits
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.baselines import CashOnlyAgent, EqualWeightAgent
from portfolio_rl.agents.ppo.factory import make_ppo_factory
from portfolio_rl.experiments.walk_forward import run_walk_forward
from portfolio_rl.experiments.reporting import make_run_id, save_json


class CloseFeature(FeatureTransformer):
    def fit(self, data):
        return None

    def transform(self, t, data):
        return data.get_bar(t).close.astype(float)


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


def make_data(seed: int = 0, steps: int = 200):
    rng = np.random.default_rng(seed)
    n_assets = 2
    base = np.linspace(10.0, 12.0, steps)[:, None]
    noise = rng.normal(scale=0.2, size=(steps, n_assets))
    close = base + np.array([[0.0, 1.0]]) + noise

    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)

    return ArrayDataSource(open=open_, high=high, low=low, close=close, volume=volume)


def format_row(name, split_idx, result):
    m = result["metrics"]
    return (
        f"{name:>12} | split {split_idx:>2} | "
        f"ret {m['cumulative_return']:+.3f} | "
        f"dd {m['max_drawdown']:.3f} | "
        f"sharpe {m['sharpe_ratio']:.2f} | "
        f"sortino {m['sortino_ratio']:.2f} | "
        f"calmar {m['calmar_ratio']:.2f} | "
        f"turn {result['mean_turnover']:.4f} | "
        f"tcost {result['mean_transaction_cost']:.4f} | "
        f"gross {result['mean_gross_exposure']:.3f}"
    )


def format_summary(name, summary):
    m = summary["metrics_mean"]
    f = summary["fields_mean"]
    ci_sharpe = m.get("ci_sharpe", (float("nan"), float("nan")))
    ci_cum = m.get("ci_cumulative_return", (float("nan"), float("nan")))
    return (
        f"{name:>12} | summary  | "
        f"ret {m.get('cumulative_return', 0.0):+.3f} "
        f"[{ci_cum[0]:+.3f},{ci_cum[1]:+.3f}] | "
        f"dd {m.get('max_drawdown', 0.0):.3f} | "
        f"sharpe {m.get('sharpe_ratio', 0.0):.2f} "
        f"[{ci_sharpe[0]:+.2f},{ci_sharpe[1]:+.2f}] | "
        f"sortino {m.get('sortino_ratio', 0.0):.2f} | "
        f"calmar {m.get('calmar_ratio', 0.0):.2f} | "
        f"turn {f.get('mean_turnover', 0.0):.4f} | "
        f"tcost {f.get('mean_transaction_cost', 0.0):.4f} | "
        f"gross {f.get('mean_gross_exposure', 0.0):.3f}"
    )


def _append_manifest(path: Path, entry: dict) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    else:
        data = []

    data.append(entry)
    save_json(path, data)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--test-size", type=int, default=40)
    parser.add_argument("--step-size", type=int, default=40)
    parser.add_argument("--embargo", type=int, default=5)
    args = parser.parse_args()

    data = make_data(seed=args.seed, steps=args.n_steps)
    splits = walk_forward_splits(
        n_steps=data.n_steps,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        embargo=args.embargo,
    )

    agents = {
        "Cash": lambda seed: CashOnlyAgent(),
        "Equal": lambda seed: EqualWeightAgent(),
        "PPO": make_ppo_factory(num_iterations=2, rollout_length=10),
    }

    print("Agent        | split | ret    | dd    | sharpe | sortino | calmar | turn   | tcost  | gross")
    print("-" * 110)

    for name, factory in agents.items():
        out_dir = None
        run_id = None
        if args.out_dir is not None:
            run_id = make_run_id(prefix=name.lower(), seed=args.seed, config={
                "n_steps": args.n_steps,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step_size": args.step_size,
                "embargo": args.embargo,
            })
            out_dir = str(Path(args.out_dir) / name.lower() / run_id)

        results = run_walk_forward(
            data_source=data,
            make_env_fn=make_env,
            agent_factory=factory,
            seed=args.seed,
            splits=splits,
            max_steps=args.test_size,
            out_dir=out_dir,
            run_metadata={
                "agent": name,
                "seed": args.seed,
                "n_steps": args.n_steps,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "step_size": args.step_size,
                "embargo": args.embargo,
            },
        )

        for idx, split in enumerate(results["splits"], start=1):
            print(format_row(name, idx, split["result"]))
        print(format_summary(name, results["summary"]))
        print("-" * 110)

        if args.out_dir is not None and run_id is not None:
            run_path = Path(args.out_dir) / name.lower() / run_id
            summary = results["summary"]["metrics_mean"]
            fields = results["summary"]["fields_mean"]
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_name": name,
                "run_id": run_id,
                "path": str(run_path),
                "cumulative_return": summary.get("cumulative_return"),
                "sharpe_ratio": summary.get("sharpe_ratio"),
                "sortino_ratio": summary.get("sortino_ratio"),
                "calmar_ratio": summary.get("calmar_ratio"),
                "mean_turnover": fields.get("mean_turnover"),
                "mean_transaction_cost": fields.get("mean_transaction_cost"),
                "mean_gross_exposure": fields.get("mean_gross_exposure"),
                "ci_sharpe": summary.get("ci_sharpe"),
                "ci_cumulative_return": summary.get("ci_cumulative_return"),
            }
            manifest_path = Path(args.out_dir) / "manifest.json"
            _append_manifest(manifest_path, entry)
            print(f"Saved report to {run_path}")


if __name__ == "__main__":
    main()
