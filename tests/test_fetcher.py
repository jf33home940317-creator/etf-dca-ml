import pandas as pd
from unittest.mock import patch, MagicMock
from data.fetcher import fetch_symbol

def test_fetch_symbol_uses_auto_adjust():
    fake = pd.DataFrame({
        "Open": [100.0], "High": [101.0], "Low": [99.0],
        "Close": [100.5], "Volume": [1000],
    }, index=pd.DatetimeIndex(["2024-01-02"]))
    with patch("data.fetcher.yf.download", return_value=fake) as mock:
        df = fetch_symbol("SPY", "2024-01-01")
    args, kwargs = mock.call_args
    assert kwargs["auto_adjust"] is True
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
