import pandas as pd
from live.ledger_settle import resolve_fill


def test_resolve_fill_returns_next_open_when_available():
    idx = pd.DatetimeIndex(["2024-01-10", "2024-01-11", "2024-01-12"])
    prices = pd.DataFrame({"open": [100.0, 101.0, 102.0]}, index=idx)
    fill = resolve_fill(signal_date=pd.Timestamp("2024-01-10"), prices=prices)
    # Signal on Jan 10 fills at Jan 10 open (live daemon runs morning of Jan 10,
    # before market open; fill is same-day open)
    assert fill == 100.0


def test_resolve_fill_returns_none_when_bar_missing():
    idx = pd.DatetimeIndex(["2024-01-10"])
    prices = pd.DataFrame({"open": [100.0]}, index=idx)
    fill = resolve_fill(signal_date=pd.Timestamp("2024-01-11"), prices=prices)
    assert fill is None
