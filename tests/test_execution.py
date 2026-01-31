import numpy as np
import pytest

from portfolio_rl.config import ExecutionConfig
from portfolio_rl.data import BarData
from portfolio_rl.execution import SimpleExecutionModel


def make_bar(price):
    price = np.array(price, dtype=float)
    return BarData(
        open=price.copy(),
        high=price.copy(),
        low=price.copy(),
        close=price.copy(),
        volume=np.ones_like(price),
        timestamp=None,
    )


def test_zero_trade_costs_and_effective_price():
    config = ExecutionConfig(commission_rate=0.001, slippage_rate=0.002, price_field="close")
    model = SimpleExecutionModel(config=config)
    bar = make_bar([10.0, 20.0])

    trade = np.array([0.0, 0.0])
    result = model.execute(trade, bar)

    assert result.total_cost == pytest.approx(0.0)
    assert result.commission == pytest.approx(0.0)
    assert result.slippage == pytest.approx(0.0)
    assert np.allclose(result.effective_price, bar.close)


def test_small_trade_costs_scale_with_notional():
    config = ExecutionConfig(commission_rate=0.001, slippage_rate=0.002, price_field="close")
    model = SimpleExecutionModel(config=config)
    bar = make_bar([10.0, 20.0])

    trade = np.array([1.0, -2.0])
    result = model.execute(trade, bar)

    notional = np.abs(trade * bar.close).sum()
    expected_commission = notional * config.commission_rate
    expected_slippage = notional * config.slippage_rate

    assert result.commission == pytest.approx(expected_commission)
    assert result.slippage == pytest.approx(expected_slippage)
    assert result.total_cost == pytest.approx(expected_commission + expected_slippage)

    expected_effective = bar.close + np.sign(trade) * bar.close * config.slippage_rate
    assert np.allclose(result.effective_price, expected_effective)


def test_large_trade_costs_scale_linearly():
    config = ExecutionConfig(commission_rate=0.001, slippage_rate=0.002, price_field="close")
    model = SimpleExecutionModel(config=config)
    bar = make_bar([10.0, 20.0])

    small_trade = np.array([1.0, -2.0])
    large_trade = small_trade * 1000.0

    small = model.execute(small_trade, bar)
    large = model.execute(large_trade, bar)

    assert large.total_cost == pytest.approx(small.total_cost * 1000.0)
    assert large.commission == pytest.approx(small.commission * 1000.0)
    assert large.slippage == pytest.approx(small.slippage * 1000.0)


def test_varying_commission_and_slippage_rates():
    bar = make_bar([50.0])
    trade = np.array([3.0])
    notional = np.abs(trade * bar.close).sum()

    config_low = ExecutionConfig(commission_rate=0.0005, slippage_rate=0.001, price_field="close")
    config_high = ExecutionConfig(commission_rate=0.002, slippage_rate=0.005, price_field="close")

    low = SimpleExecutionModel(config=config_low).execute(trade, bar)
    high = SimpleExecutionModel(config=config_high).execute(trade, bar)

    assert low.total_cost == pytest.approx(notional * (config_low.commission_rate + config_low.slippage_rate))
    assert high.total_cost == pytest.approx(notional * (config_high.commission_rate + config_high.slippage_rate))
    assert high.total_cost > low.total_cost
