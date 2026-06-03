import numpy as np
import pandas as pd
from features.builder import build_features_for_symbol


def test_builder_combines_tech_macro_labels():
    idx = pd.bdate_range("2023-01-01", periods=400, name="date")
    close = pd.Series(
        100 + np.linspace(0, 10, 400) + np.sin(np.arange(400) / 5.0),
        index=idx,
    )
    sym_df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 1e6}, index=idx)
    macro_df = pd.DataFrame({
        "vix": 20.0, "vix_ma200_ratio": 1.0, "vix_chg5": 0.0,
        "tnx": 4.0, "tnx_slope": 0.0,
        "dxy": 100.0, "dxy_chg5": 0.0, "smh_roc1": 0.0,
    }, index=idx)
    out = build_features_for_symbol(sym_df, macro_df)
    assert "rsi14" in out.columns
    assert "vix" in out.columns
    assert "label" in out.columns
    assert out.dropna(subset=["rsi14", "vix"]).shape[0] > 0
