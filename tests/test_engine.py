import numpy as np
import pandas as pd
from backtest.engine import run_backtest

class DummyModel:
    def __init__(self, prob_series): self.prob_series = prob_series
    def predict_proba(self, X):
        # X is feature matrix; we look up by its index
        idx = X.index
        p = self.prob_series.reindex(idx).fillna(0.0).values
        return np.column_stack([1 - p, p])

def _make_price_frame():
    idx = pd.bdate_range("2024-01-01", "2024-03-29", name="date")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    return pd.DataFrame({"open": close, "high": close + 1,
                         "low": close - 1, "close": close,
                         "volume": 1e6}, index=idx)

def test_backtest_fires_when_prob_above_threshold():
    prices = _make_price_frame()
    feat = pd.DataFrame({"x": 1.0}, index=prices.index)
    # Make prob high on Jan 10 only
    probs = pd.Series(0.0, index=prices.index)
    probs.loc["2024-01-10"] = 0.9
    model = DummyModel(probs)
    trades = run_backtest(
        symbol="TEST", prices=prices, features=feat, model=model,
        feature_cols=["x"], monthly_budget=1000.0,
        threshold=0.55, max_per_trade_ratio=0.25,
    )
    buys = [t for t in trades if t["reason"] == "BUY"]
    assert len(buys) == 1
    assert buys[0]["date"] == pd.Timestamp("2024-01-10")
    assert buys[0]["amount"] == 250.0  # B/4

def test_backtest_forces_buy_on_last_trading_day():
    prices = _make_price_frame()
    feat = pd.DataFrame({"x": 1.0}, index=prices.index)
    probs = pd.Series(0.0, index=prices.index)
    model = DummyModel(probs)
    trades = run_backtest(
        symbol="TEST", prices=prices, features=feat, model=model,
        feature_cols=["x"], monthly_budget=1000.0,
        threshold=0.55, max_per_trade_ratio=0.25,
    )
    forced = [t for t in trades if t["reason"] == "FORCED_BUY"]
    # 3 months in window, each should force-buy 1000 at month-end
    assert len(forced) == 3
    assert all(t["amount"] == 1000.0 for t in forced)
