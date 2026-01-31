import numpy as np
import pytest

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import Action, PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward


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


def make_env(data, execution_config, initial_cash=1000.0, cost_weight=1.0):
    features = FeaturePipeline([CloseFeature()])
    execution_model = SimpleExecutionModel(config=execution_config)
    reward_fn = LogReturnReward(cost_weight=cost_weight)
    config = EnvConfig(initial_cash=initial_cash)
    return PortfolioEnv(
        data=data,
        features=features,
        execution_model=execution_model,
        reward_fn=reward_fn,
        config=config,
    )


def compute_trade_quantity(state_positions, prev_value, weights, price):
    target_value = prev_value * weights
    target_positions = target_value / price
    return target_positions - state_positions


def expected_cost(trade_quantity, price, commission_rate, slippage_rate):
    notional = np.abs(trade_quantity * price).sum()
    return notional * (commission_rate + slippage_rate)


def test_environment_cost_accounting_and_reward_consistency():
    # Tiny synthetic 2-asset dataset, 4 timesteps
    close = [
        [10.0, 20.0],
        [11.0, 18.0],
        [12.0, 21.0],
        [13.0, 19.0],
    ]
    data = make_data(close)

    exec_cfg = ExecutionConfig(commission_rate=0.001, slippage_rate=0.002, price_field="close")
    env = make_env(data, execution_config=exec_cfg, initial_cash=1000.0, cost_weight=1.0)

    obs = env.reset()
    price_t0 = obs.prices

    # (a) initial allocation
    action_a = Action(weights=np.array([0.6, 0.4]))
    step_a = env.step(action_a)

    prev_value_a = step_a.info["prev_value"]
    trade_qty_a = compute_trade_quantity(np.zeros(2), prev_value_a, action_a.weights, price_t0)
    expected_cost_a = expected_cost(trade_qty_a, price_t0, exec_cfg.commission_rate, exec_cfg.slippage_rate)

    assert step_a.info["transaction_cost"] == pytest.approx(expected_cost_a)
    for key in ["cash", "cash_frac", "gross_exposure", "net_exposure", "weights_sum", "turnover"]:
        assert key in step_a.info
        assert np.isfinite(step_a.info[key])
    assert step_a.info["cash"] < 0.0
    assert step_a.info["weights_sum"] > 1.0
    assert step_a.info["gross_exposure"] >= 0.0
    assert abs(step_a.info["net_exposure"]) <= step_a.info["gross_exposure"] + 1e-12

    next_price_a = data.get_bar(1).close
    next_value_a = step_a.info["next_value"]
    assert next_value_a == pytest.approx(step_a.observation.cash + np.dot(step_a.observation.positions, next_price_a))

    expected_reward_a = np.log(next_value_a / prev_value_a) - (expected_cost_a / prev_value_a)
    assert step_a.reward == pytest.approx(expected_reward_a)

    # (b) rebalance that changes weights
    price_t1 = data.get_bar(1).close
    action_b = Action(weights=np.array([0.2, 0.8]))
    step_b = env.step(action_b)

    prev_value_b = step_b.info["prev_value"]
    trade_qty_b = compute_trade_quantity(step_a.observation.positions, prev_value_b, action_b.weights, price_t1)
    expected_cost_b = expected_cost(trade_qty_b, price_t1, exec_cfg.commission_rate, exec_cfg.slippage_rate)

    assert step_b.info["transaction_cost"] == pytest.approx(expected_cost_b)
    for key in ["cash", "cash_frac", "gross_exposure", "net_exposure", "weights_sum", "turnover"]:
        assert key in step_b.info
        assert np.isfinite(step_b.info[key])
    assert step_b.info["gross_exposure"] >= 0.0
    assert abs(step_b.info["net_exposure"]) <= step_b.info["gross_exposure"] + 1e-12

    next_price_b = data.get_bar(2).close
    next_value_b = step_b.info["next_value"]
    assert next_value_b == pytest.approx(step_b.observation.cash + np.dot(step_b.observation.positions, next_price_b))

    expected_reward_b = np.log(next_value_b / prev_value_b) - (expected_cost_b / prev_value_b)
    assert step_b.reward == pytest.approx(expected_reward_b)

    # (c) no-trade action (weights aligned with current holdings)
    price_t2 = data.get_bar(2).close
    current_value = step_b.observation.cash + np.dot(step_b.observation.positions, price_t2)
    current_weights = (step_b.observation.positions * price_t2) / current_value

    action_c = Action(weights=current_weights)
    step_c = env.step(action_c)

    assert step_c.info["transaction_cost"] == pytest.approx(0.0)
    assert step_c.info["turnover"] == pytest.approx(0.0)
    for key in ["cash", "cash_frac", "gross_exposure", "net_exposure", "weights_sum", "turnover"]:
        assert key in step_c.info
        assert np.isfinite(step_c.info[key])
    assert step_c.info["gross_exposure"] >= 0.0
    assert abs(step_c.info["net_exposure"]) <= step_c.info["gross_exposure"] + 1e-12

    next_price_c = data.get_bar(3).close
    next_value_c = step_c.info["next_value"]
    assert next_value_c == pytest.approx(step_c.observation.cash + np.dot(step_c.observation.positions, next_price_c))

    expected_reward_c = np.log(next_value_c / step_c.info["prev_value"]) - (0.0 / step_c.info["prev_value"])
    assert step_c.reward == pytest.approx(expected_reward_c)
