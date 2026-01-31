from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from portfolio_rl.eval.rollout import run_episode
from portfolio_rl.eval.metrics import (
    cumulative_return,
    max_drawdown,
    realized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)


def _simple_returns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.array([], dtype=float)
    return values[1:] / values[:-1] - 1.0


def run_experiment(env, agent, seed: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run an evaluation episode and compute metrics.

    Uses simple returns: r_t = V_t / V_{t-1} - 1.
    """

    _ = env.reset(seed=seed)
    results = run_episode(env, agent, max_steps=max_steps)

    values = results["portfolio_values"]
    returns = _simple_returns(values)

    metrics = {
        "cumulative_return": cumulative_return(values),
        "max_drawdown": max_drawdown(values),
        "realized_volatility": realized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "calmar_ratio": calmar_ratio(values),
    }

    return {
        "metrics": metrics,
        "mean_transaction_cost": float(np.mean(results["transaction_costs"])) if results["transaction_costs"].size else 0.0,
        "mean_turnover": float(np.mean(results["turnover"])) if results["turnover"].size else 0.0,
        "mean_gross_exposure": float(np.mean(results["gross_exposure"])) if results["gross_exposure"].size else 0.0,
        "mean_net_exposure": float(np.mean(results["net_exposure"])) if results["net_exposure"].size else 0.0,
        "mean_weights_sum": float(np.mean(results["weights_sum"])) if results["weights_sum"].size else 0.0,
        "final_cash": float(results["cash"][-1]) if results["cash"].size else 0.0,
        "final_portfolio_value": float(results["portfolio_values"][-1]) if results["portfolio_values"].size else 0.0,
        "last_info": results.get("last_info", {}),
    }
