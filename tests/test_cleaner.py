import pandas as pd
from data.cleaner import drop_invalid, is_last_trading_day_of_month, align_macro

def test_drop_invalid_removes_zero_volume():
    df = pd.DataFrame({
        "open": [10, 10], "high": [10, 10], "low": [10, 10],
        "close": [10, 10], "volume": [1000, 0],
    }, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date"))
    out = drop_invalid(df)
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-02")

def test_last_trading_day_true_on_month_end_session():
    idx = pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"])
    df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=idx)
    assert is_last_trading_day_of_month(df, pd.Timestamp("2024-01-31")) is True
    assert is_last_trading_day_of_month(df, pd.Timestamp("2024-01-30")) is False

def test_last_trading_day_handles_holiday_close():
    # TW closes for CNY: last Jan session is Jan 29, no Jan 30/31 bars
    idx = pd.DatetimeIndex(["2024-01-26", "2024-01-29", "2024-02-15"])
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    assert is_last_trading_day_of_month(df, pd.Timestamp("2024-01-29")) is True

def test_align_macro_backward_fill():
    sym = pd.DataFrame({"close": [1, 2, 3]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="date"))
    macro = pd.DataFrame({"vix": [20.0]},
        index=pd.DatetimeIndex(["2024-01-02"], name="date"))
    out = align_macro(sym, macro)
    assert out["vix"].tolist() == [20.0, 20.0, 20.0]
