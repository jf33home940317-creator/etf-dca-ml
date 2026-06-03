import numpy as np
import pandas as pd
from features.labels import compute_dca_labels


def test_label_top_quintile_in_month():
    idx = pd.bdate_range("2024-01-01", "2024-03-31", name="date")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, len(idx))) + np.arange(len(idx)) * 0.1,
                      index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume": 1_000_000}, index=idx)
    out = compute_dca_labels(df, window=20, top_q=0.20)
    assert "label" in out.columns
    # Only completed months with full forward window keep labels
    feb = out.loc["2024-02", "label"].dropna()
    pos_rate = feb.mean()
    # top 20% means roughly 0.2 of valid rows in Feb labeled 1
    assert 0.05 <= pos_rate <= 0.35


def test_label_drops_tail_with_incomplete_forward_window():
    idx = pd.bdate_range("2024-01-01", "2024-02-29", name="date")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": 1e6}, index=idx)
    out = compute_dca_labels(df, window=20, top_q=0.20)
    # Last 20 rows must have NaN forward_return → label NaN
    assert out["label"].iloc[-20:].isna().all()
