import numpy as np
import pytest

from portfolio_rl.config import EnvConfig, ExecutionConfig
from portfolio_rl.data import ArrayDataSource
from portfolio_rl.env import Action, PortfolioEnv
from portfolio_rl.execution import SimpleExecutionModel
from portfolio_rl.features import FeaturePipeline, FeatureTransformer
from portfolio_rl.reward import LogReturnReward


class CloseFeature(FeatureTransformer):
    """Return current close prices as features."""

    def fit(self, data):
        return None

    def transform(self, t, data):
        bar = data.get_bar(t)
        return bar.close.astype(float)


class RewardSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, prev_value: float, next_value: float, transaction_cost: float) -> float:
        self.calls.append((prev_value, next_value, transaction_cost))
        return 0.0


def make_data(close_values):
    close = np.array(close_values, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)
    return ArrayDataSource(open=open_, high=high, low=low, close=close, volume=volume)


def make_env(data, reward_fn=None, execution_config=None, initial_cash=100.0):
    reward_fn = reward_fn or LogReturnReward()
    execution_config = execution_config or ExecutionConfig()
    features = FeaturePipeline([CloseFeature()])
    execution_model = SimpleExecutionModel(config=execution_config)
    config = EnvConfig(initial_cash=initial_cash)
    return PortfolioEnv(
        data=data,
        features=features,
        execution_model=execution_model,
        reward_fn=reward_fn,
        config=config,
    )


def test_observation_does_not_depend_on_t_plus_one_prices():
    data_a = make_data([[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]])
    data_b = make_data([[10.0, 12.0], [999.0, 999.0], [30.0, 32.0]])

    env_a = make_env(data_a)
    env_b = make_env(data_b)

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    assert np.allclose(obs_a.features, obs_b.features)
    assert np.allclose(obs_a.prices, obs_b.prices)
    assert np.allclose(obs_a.positions, obs_b.positions)
    assert obs_a.cash == obs_b.cash


def test_reward_uses_t_plus_one_mark_to_market():
    data = make_data([10.0, 20.0])
    spy = RewardSpy()
    execution_config = ExecutionConfig(commission_rate=0.0, slippage_rate=0.0)
    env = make_env(data, reward_fn=spy, execution_config=execution_config, initial_cash=100.0)

    env.reset()
    env.step(Action(weights=np.array([1.0])))

    assert len(spy.calls) == 1
    prev_value, next_value, transaction_cost = spy.calls[0]

    assert prev_value == pytest.approx(100.0)
    assert next_value == pytest.approx(200.0)
    assert transaction_cost == pytest.approx(0.0)


def test_transaction_costs_apply_on_trade_and_zero_on_no_trade():
    data = make_data([[10.0], [10.0], [10.0]])
    execution_config = ExecutionConfig(commission_rate=0.001, slippage_rate=0.0)
    env = make_env(data, execution_config=execution_config, initial_cash=100.0)

    env.reset()
    no_trade = env.step(Action(weights=np.array([0.0])))
    assert no_trade.info["transaction_cost"] == pytest.approx(0.0)

    trade = env.step(Action(weights=np.array([1.0])))
    assert trade.info["transaction_cost"] > 0.0


def test_action_validation_rejects_non_finite_and_shape_mismatch():
    data = make_data([[10.0, 10.0], [10.0, 10.0]])
    env = make_env(data)

    env.reset()
    with pytest.raises(ValueError):
        env.step(Action(weights=np.array([1.0, 2.0, 3.0])))

    env.reset()
    with pytest.raises(ValueError):
        env.step(Action(weights=np.array([np.nan, 1.0])))


def test_seeding_makes_rollouts_deterministic():
    data = make_data([[10.0, 12.0], [11.0, 13.0], [12.0, 14.0]])

    env_a = make_env(data)
    env_b = make_env(data)

    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)

    assert np.allclose(obs_a.features, obs_b.features)
    assert np.allclose(obs_a.prices, obs_b.prices)

    actions = [Action(weights=np.array([0.5, 0.5])), Action(weights=np.array([0.2, 0.8]))]

    for action in actions:
        step_a = env_a.step(action)
        step_b = env_b.step(action)

        assert step_a.reward == pytest.approx(step_b.reward)
        assert step_a.info["transaction_cost"] == pytest.approx(step_b.info["transaction_cost"])
        assert np.allclose(step_a.observation.prices, step_b.observation.prices)
