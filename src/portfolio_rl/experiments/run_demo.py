from __future__ import annotations

import numpy as np

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward
from portfolio_rl.agents.baselines import HoldAgent, EqualWeightAgent, CashOnlyAgent
from portfolio_rl.agents.ppo.trainer import train_ppo
from portfolio_rl.agents.ppo.policy import PPOPolicy
from portfolio_rl.experiments.run import run_experiment


class CloseFeature(FeatureTransformer):
    def fit(self, data):
        return None

    def transform(self, t, data):
        return data.get_bar(t).close.astype(float)


def make_env(seed: int = 0):
    rng = np.random.default_rng(seed)
    steps = 50
    n_assets = 2
    base = np.linspace(10.0, 12.0, steps)[:, None]
    noise = rng.normal(scale=0.2, size=(steps, n_assets))
    close = base + np.array([[0.0, 1.0]]) + noise

    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)

    data = ArrayDataSource(open=open_, high=high, low=low, close=close, volume=volume)

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


def format_row(name, result):
    m = result["metrics"]
    return (
        f"{name:>12} | "
        f"ret {m['cumulative_return']:+.3f} | "
        f"dd {m['max_drawdown']:.3f} | "
        f"sharpe {m['sharpe_ratio']:.2f} | "
        f"sortino {m['sortino_ratio']:.2f} | "
        f"calmar {m['calmar_ratio']:.2f} | "
        f"tcost {result['mean_transaction_cost']:.4f} | "
        f"turn {result['mean_turnover']:.4f}"
    )


def main():
    baselines = {
        "Hold": HoldAgent(),
        "Equal": EqualWeightAgent(),
        "Cash": CashOnlyAgent(),
    }

    results = []
    for name, agent in baselines.items():
        res = run_experiment(make_env(seed=0), agent, seed=0)
        results.append((name, res))

    # Tiny PPO run
    env_ppo = make_env(seed=0)
    train_ppo(env_ppo, seed=0, num_iterations=2, rollout_length=10)

    env_eval = make_env(seed=0)
    obs = env_eval.reset(seed=0)
    obs_vec = np.concatenate([obs.features, obs.prices, obs.positions, obs.weights, np.array([obs.cash])])
    ppo_policy = PPOPolicy(obs_dim=obs_vec.shape[0], n_assets=obs.prices.shape[0], seed=0)
    ppo_result = run_experiment(make_env(seed=0), agent=ppo_policy, seed=0)
    results.append(("PPO", ppo_result))

    print("Agent        | ret    | dd    | sharpe | sortino | calmar | tcost  | turn")
    print("-" * 86)
    for name, res in results:
        print(format_row(name, res))


if __name__ == "__main__":
    main()
