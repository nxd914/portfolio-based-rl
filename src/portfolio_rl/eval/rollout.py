from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from portfolio_rl.agents.base import Agent
from portfolio_rl.env.types import Observation


def _portfolio_value(observation: Observation) -> float:
    return float(observation.cash + np.dot(observation.positions, observation.prices))


def run_episode(env, agent: Agent, max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run a single episode and collect time-series outputs.

    Returns
    -------
    dict
        rewards, portfolio_values, transaction_costs, turnover, cash as numpy arrays,
        plus last_info from the final step.
    """

    if max_steps is not None and max_steps <= 0:
        return {
            "rewards": np.array([], dtype=float),
            "portfolio_values": np.array([], dtype=float),
            "transaction_costs": np.array([], dtype=float),
            "turnover": np.array([], dtype=float),
            "cash": np.array([], dtype=float),
            "last_info": {},
        }

    obs = env.reset()

    rewards = []
    values = []
    costs = []
    turnover = []
    cash = []
    gross_exposure = []
    net_exposure = []
    weights_sum = []
    cash_frac = []
    last_info: Dict[str, Any] = {}

    steps = 0
    done = False

    while not done:
        action = agent.act(obs)
        step = env.step(action)
        obs = step.observation

        rewards.append(step.reward)
        values.append(_portfolio_value(obs))
        costs.append(step.info.get("transaction_cost", 0.0))
        turnover.append(step.info.get("turnover", 0.0))
        cash.append(step.info.get("cash", obs.cash))
        gross_exposure.append(step.info.get("gross_exposure", 0.0))
        net_exposure.append(step.info.get("net_exposure", 0.0))
        weights_sum.append(step.info.get("weights_sum", 0.0))
        cash_frac.append(step.info.get("cash_frac", 0.0))

        last_info = step.info
        steps += 1
        done = step.done or (max_steps is not None and steps >= max_steps)

    return {
        "rewards": np.asarray(rewards, dtype=float),
        "portfolio_values": np.asarray(values, dtype=float),
        "transaction_costs": np.asarray(costs, dtype=float),
        "turnover": np.asarray(turnover, dtype=float),
        "cash": np.asarray(cash, dtype=float),
        "gross_exposure": np.asarray(gross_exposure, dtype=float),
        "net_exposure": np.asarray(net_exposure, dtype=float),
        "weights_sum": np.asarray(weights_sum, dtype=float),
        "cash_frac": np.asarray(cash_frac, dtype=float),
        "last_info": last_info,
    }
