from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from portfolio_rl.config import EnvConfig
from portfolio_rl.data.base import DataSource
from portfolio_rl.execution.base import ExecutionModel
from portfolio_rl.features.pipeline import FeaturePipeline
from portfolio_rl.reward.base import RewardFn
from portfolio_rl.utils.seed import seed_all
from portfolio_rl.utils.validation import normalize_weights

from .state import PortfolioState
from .types import Action, Observation, StepResult


@dataclass
class PortfolioEnv:
    """Portfolio-based RL environment with explicit execution and costs.

    Notes
    - Actions are target asset weights; any remainder is implicitly held as cash.
    - Cash can become negative after transaction costs; when cash is negative,
      current asset weights can sum to > 1. This is an implicit leverage/budget
      violation unless an explicit budget constraint is applied.
    - Observations at time t only use data available at time t.
    - Rewards are realized at time t+1 via mark-to-market prices.
    - Trading occurs at time t using the current bar's price field.
    - Turnover is defined as sum(abs(traded_notional)) / prev_value.
    """

    data: DataSource
    features: FeaturePipeline
    execution_model: ExecutionModel
    reward_fn: RewardFn
    config: EnvConfig = EnvConfig()

    def __post_init__(self) -> None:
        self._t: int = self.config.start_t
        self._max_t: int = self.data.n_steps - 2
        if self._max_t < self.config.start_t:
            raise ValueError("data.n_steps must be at least start_t + 2")
        self._rng = None
        self._state = PortfolioState(
            cash=float(self.config.initial_cash),
            positions=np.zeros(self.data.n_assets, dtype=float),
        )

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._rng = seed_all(seed)
        self._t = self.config.start_t
        self._state = PortfolioState(
            cash=float(self.config.initial_cash),
            positions=np.zeros(self.data.n_assets, dtype=float),
        )
        return self._build_observation(self._t)

    def step(self, action: Action) -> StepResult:
        if self._t > self._max_t:
            raise RuntimeError("episode is done; call reset()")

        if action.weights.shape != (self.data.n_assets,):
            raise ValueError("action.weights must match number of assets")
        if not np.isfinite(action.weights).all():
            raise ValueError("action.weights must be finite")

        weights = action.weights
        if self.config.constraints.long_only:
            weights = normalize_weights(weights, self.config.constraints.max_weight)

        bar = self.data.get_bar(self._t)
        price = getattr(bar, self.config.execution.price_field)
        if np.any(price <= 0):
            raise ValueError("price must be positive for allocation")

        prev_value = self._state.value(price)
        target_value = prev_value * weights
        target_positions = target_value / price
        trade_quantity = target_positions - self._state.positions

        execution = self.execution_model.execute(trade_quantity, bar)

        cash_after = self._state.cash - float(np.dot(execution.filled_quantity, price)) - execution.total_cost
        positions_after = self._state.positions + execution.filled_quantity
        self._state = PortfolioState(cash=cash_after, positions=positions_after)

        asset_values = self._state.positions * price
        total_value_now = float(self._state.cash + asset_values.sum())
        if total_value_now != 0.0:
            current_weights = asset_values / total_value_now
            cash_frac = float(self._state.cash / total_value_now)
            gross_exposure = float(np.abs(asset_values).sum() / total_value_now)
            net_exposure = float(asset_values.sum() / total_value_now)
            weights_sum = float(current_weights.sum())
        else:
            current_weights = np.zeros_like(self._state.positions)
            cash_frac = 0.0
            gross_exposure = 0.0
            net_exposure = 0.0
            weights_sum = 0.0

        traded_notional = float(np.abs(trade_quantity * price).sum())
        turnover = traded_notional / prev_value if prev_value != 0.0 else 0.0

        next_bar = self.data.get_bar(self._t + 1)
        next_price = getattr(next_bar, self.config.execution.price_field)
        next_value = self._state.value(next_price)

        reward = self.reward_fn(prev_value, next_value, execution.total_cost)

        self._t += 1
        done = self._t >= self._max_t + 1

        obs = self._build_observation(self._t)

        info = {
            "prev_value": prev_value,
            "next_value": next_value,
            "transaction_cost": execution.total_cost,
            "commission": execution.commission,
            "slippage": execution.slippage,
            "cash": float(self._state.cash),
            "cash_frac": float(cash_frac),
            "gross_exposure": float(gross_exposure),
            "net_exposure": float(net_exposure),
            "weights_sum": float(weights_sum),
            "turnover": float(turnover),
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _build_observation(self, t: int) -> Observation:
        bar = self.data.get_bar(t)
        price = getattr(bar, self.config.execution.price_field)
        features = self.features.transform(t, self.data)
        weights = self._state.weights(price)

        return Observation(
            features=features,
            prices=price,
            positions=self._state.positions.copy(),
            cash=self._state.cash,
            weights=weights,
        )
