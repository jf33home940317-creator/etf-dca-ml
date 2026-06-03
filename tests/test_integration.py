"""End-to-end: fetch → features → train → backtest. Network required.
Marked as integration; skip in fast CI."""
import pytest
import config
from data.fetcher import fetch_symbol
from data.cleaner import drop_invalid
from features.macro import build_macro_frame
from features.builder import build_features_for_symbol, FEATURE_COLS
from models.trainer import train_symbol
from backtest.engine import run_backtest


@pytest.mark.integration
def test_full_pipeline_spy_small_window():
    sym = drop_invalid(fetch_symbol("SPY", start="2020-01-01"))
    macro_raw = {
        "VIX": drop_invalid(fetch_symbol("^VIX", start="2020-01-01")),
        "TNX": drop_invalid(fetch_symbol("^TNX", start="2020-01-01")),
        "DXY": drop_invalid(fetch_symbol("DX-Y.NYB", start="2020-01-01")),
        "SMH": drop_invalid(fetch_symbol("SMH", start="2020-01-01")),
    }
    macro_df = build_macro_frame(macro_raw)
    feat = build_features_for_symbol(sym, macro_df)
    model, meta = train_symbol(feat, FEATURE_COLS)
    assert meta["n_samples"] > 100
    trades = run_backtest(
        symbol="SPY", prices=sym, features=feat, model=model,
        feature_cols=FEATURE_COLS, monthly_budget=1000.0,
        threshold=0.55, max_per_trade_ratio=0.25,
    )
    # At least some forced buys (months with no signal)
    assert any(t["reason"] == "FORCED_BUY" for t in trades)
