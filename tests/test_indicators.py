import numpy as np
import pandas as pd
from features.indicators import compute_indicators

def test_compute_indicators_produces_expected_columns():
    n = 300
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2023-01-01", periods=n, name="date")
    close = pd.Series(np.abs(100 + np.cumsum(rng.normal(0, 1, n))) + 50, index=idx)
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1e6, 1e7, n),
    }, index=idx)
    out = compute_indicators(df)
    for col in [
        "close_ma20", "close_ma60", "ma20_ma60",
        "rsi14", "macd_hist", "roc5", "roc20",
        "atr_norm", "bb_width20", "rvol20",
        "channel_pos60", "dd_60", "dd_252",
        "vol_norm", "obv_slope",
        "day_of_month", "day_of_week",
    ]:
        assert col in out.columns, f"missing {col}"
    assert out.dropna().shape[0] > 0
