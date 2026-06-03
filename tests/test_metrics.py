import numpy as np
import pandas as pd
from backtest.metrics import (
    equity_curve_from_trades,
    classic_dca_baseline,
    weekly_dca_baseline,
    summary_stats,
)


def _prices():
    idx = pd.bdate_range("2024-01-01", "2024-06-28", name="date")
    return pd.DataFrame({
        "open": np.linspace(100, 130, len(idx)),
        "close": np.linspace(100, 130, len(idx)),
        "high": np.linspace(101, 131, len(idx)),
        "low": np.linspace(99, 129, len(idx)),
        "volume": 1e6,
    }, index=idx)


def test_classic_dca_buys_on_first_session_of_each_month():
    prices = _prices()
    trades = classic_dca_baseline("X", prices, monthly_budget=1000.0)
    assert len(trades) == 6
    assert trades[0]["reason"] == "CLASSIC_DCA"


def test_weekly_dca_buys_each_monday():
    prices = _prices()
    trades = weekly_dca_baseline("X", prices, monthly_budget=1000.0)
    # ~26 Mondays in 6 months
    assert 20 <= len(trades) <= 30


def test_equity_curve_and_summary():
    prices = _prices()
    trades = classic_dca_baseline("X", prices, monthly_budget=1000.0)
    eq = equity_curve_from_trades(trades, prices)
    assert eq.iloc[-1] > 0
    stats = summary_stats(eq)
    for k in ["total_return", "cagr", "sharpe", "max_dd", "calmar"]:
        assert k in stats
