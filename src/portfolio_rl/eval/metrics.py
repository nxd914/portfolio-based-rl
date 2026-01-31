from __future__ import annotations

import numpy as np


def cumulative_return(values: np.ndarray) -> float:
    """Compute cumulative return from a series of portfolio values.

    Assumes values are positive portfolio values over time.
    Uses simple return: (final / initial) - 1.
    """

    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    initial = values[0]
    final = values[-1]
    if initial == 0.0:
        return 0.0
    return float(final / initial - 1.0)


def max_drawdown(values: np.ndarray) -> float:
    """Compute maximum drawdown from a series of portfolio values.

    Returns the maximum peak-to-trough decline as a positive number.
    """

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0

    peaks = np.maximum.accumulate(values)
    drawdowns = np.zeros_like(values, dtype=float)
    positive_peaks = peaks > 0.0
    drawdowns[positive_peaks] = (peaks[positive_peaks] - values[positive_peaks]) / peaks[positive_peaks]
    return float(np.max(drawdowns))


def realized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute annualized realized volatility from simple returns.

    Uses population standard deviation (ddof=0). Returns 0 for empty input.
    """

    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns, ddof=0))
    if std == 0.0:
        return 0.0
    return std * float(np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from simple returns.

    Assumes risk_free_rate is annualized and converts it to per-period.
    Returns 0 if standard deviation is zero or input is empty.
    """

    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0

    rf_per_period = risk_free_rate / float(periods_per_year)
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=0))
    if std == 0.0:
        return 0.0
    mean = float(np.mean(excess))
    return mean / std * float(np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio from simple returns.

    Downside deviation uses only returns below the per-period risk-free rate.
    Returns 0.0 if input is empty or downside deviation is zero.
    """

    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0

    rf_per_period = risk_free_rate / float(periods_per_year)
    excess = returns - rf_per_period
    downside = excess[excess < 0.0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.std(downside, ddof=0))
    if downside_std == 0.0:
        return 0.0
    mean_excess = float(np.mean(excess))
    return mean_excess / downside_std * float(np.sqrt(periods_per_year))


def calmar_ratio(values: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute Calmar ratio using annualized return over max drawdown.

    Annualized return uses: (final / initial) ** (periods_per_year / (n-1)) - 1.
    Returns 0.0 if input is empty or max drawdown is zero.
    """

    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    if values[0] == 0.0:
        return 0.0

    n = values.size
    cumulative = float(values[-1] / values[0])
    ann_return = cumulative ** (periods_per_year / float(n - 1)) - 1.0

    dd = max_drawdown(values)
    if dd == 0.0:
        return 0.0
    return ann_return / dd
