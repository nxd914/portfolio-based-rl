import numpy as np
import pytest

from portfolio_rl.data import ArrayDataSource
from portfolio_rl.features.basic import CausalReturns, RollingVolatility


def make_data(close_values):
    close = np.array(close_values, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    open_ = close.copy()
    high = close.copy()
    low = close.copy()
    volume = np.ones_like(close)
    return ArrayDataSource(open=open_, high=high, low=low, close=close, volume=volume)


def test_leakage_safety_returns_and_vol():
    base = [
        [10.0, 20.0],
        [11.0, 22.0],
        [12.0, 24.0],
        [13.0, 26.0],
    ]
    altered = [
        [10.0, 20.0],
        [11.0, 22.0],
        [12.0, 24.0],
        [999.0, 999.0],
    ]

    data_a = make_data(base)
    data_b = make_data(altered)

    returns = CausalReturns(window=1)
    vol = RollingVolatility(window=2)

    t = 2
    out_a_r = returns.transform(t, data_a)
    out_b_r = returns.transform(t, data_b)
    out_a_v = vol.transform(t, data_a)
    out_b_v = vol.transform(t, data_b)

    assert np.allclose(out_a_r, out_b_r)
    assert np.allclose(out_a_v, out_b_v)


def test_correctness_and_edge_cases():
    prices = [100.0, 110.0, 121.0]
    data = make_data(prices)

    returns = CausalReturns(window=1)
    vol = RollingVolatility(window=2)

    # Edge cases: insufficient history
    assert np.allclose(returns.transform(0, data), np.zeros(1))
    assert np.allclose(vol.transform(1, data), np.zeros(1))

    # Correctness at t=2
    ret_t2 = returns.transform(2, data)[0]
    assert ret_t2 == pytest.approx(0.1)

    # Rolling vol over returns at t=2: [0.1, 0.1] -> std 0
    vol_t2 = vol.transform(2, data)[0]
    assert vol_t2 == pytest.approx(0.0)
